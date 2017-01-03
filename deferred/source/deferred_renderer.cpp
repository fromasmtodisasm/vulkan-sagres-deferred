#include <deferred_renderer.h>
#include <base_system.h>
#include <vulkan_device.h>
#include <array>
#include <vulkan_tools.h>
#include <model.h>
#include <cassert>
#include <logger.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <material_constants.h>
#include <camera.h>
#include <glm/gtc/type_ptr.hpp>
#include <material_texture_type.h>
#include <vertex_setup.h>
#include <EASTL/vector.h>
#include <random>
#include <cstring>
#include <vulkan_texture.h>
#include <vulkan_image.h>
#include <meshes_heap_manager.h>

namespace vks {

extern const VkFormat kColourBufferFormat = VK_FORMAT_B8G8R8A8_SRGB;
const VkFormat kDiffuseAlbedoFormat = VK_FORMAT_R8G8B8A8_UNORM;
const VkFormat kSpecularAlbedoFormat = VK_FORMAT_R8G8B8A8_UNORM;
const VkFormat kNormalFormat = VK_FORMAT_R16G16B16A16_SFLOAT;
const VkFormat kPositionFormat = VK_FORMAT_R16G16B16A16_SFLOAT;
const VkFormat kAccumulationFormat = VK_FORMAT_R16G16B16A16_SFLOAT;
const uint32_t kProjViewMatricesBindingPos = 0U;
const uint32_t kDepthBufferBindingPos = 2U;
const uint32_t kGBufferBaseBindingPos = 10U;
const uint32_t kSpecInfoDrawCmdsCountID = 0U;
const uint32_t kUniformBufferDescCount = 5U;
const uint32_t kSetsCount = 3U;
const uint32_t kBindingsCount = 10U;
const uint32_t kMainStaticBuffBindingPos = 0U;
const uint32_t kLightsArrayBindingPos = 8U;
const uint32_t kMatConstsArrayBindingPos = 9U;
const uint32_t kDepthBuffBindingPos = 1U;
const uint32_t kDiffuseTexturesArrayBindingPos = 2U;
const uint32_t kAmbientTexturesArrayBindingPos = 3U;
const uint32_t kSpecularTexturesArrayBindingPos = 4U;
const uint32_t kNormalTexturesArrayBindingPos = 5U;
const uint32_t kRoughnessTexturesArrayBindingPos = 6U;
const uint32_t kAccumulationBufferBindingPos = 7U;
const uint32_t kMaxNumUniformBuffers = 5U;
const uint32_t kMaxNumSSBOs = 30U;
const uint32_t kMaxNumMatInstances = 30U;
const uint32_t kNumMeshesSpecConstPos = 0U;
const uint32_t kNumMaterialsSpecConstPos = 0U;
const uint32_t kNumIndirectDrawsSpecConstPos = 1U;
const uint32_t kNumLightsSpecConstPos = 1U;
extern const uint32_t kVertexBuffersBaseBindPos;
extern const uint32_t kIndirectDrawCmdsBindingPos;
extern const uint32_t kIdxBufferBindPos;
extern const uint32_t kModelMatxsBufferBindPos;
extern const uint32_t kMaterialIDsBufferBindPos;

//const uint32_t kIndirectDrawCmdsBindingPos = 4U;
const uint32_t kSSAOKernelSize = 64U;
const uint32_t kNoiseTextureSize = 16U;

DeferredRenderer::DeferredRenderer()
  : renderpass_(),
  framebuffers_(),
  cmd_buffers_(),
  g_buffer_(),
  accum_buffer_(),
  depth_buffer_(),
  depth_buffer_depth_view_(nullptr),
  g_store_material_(),
  g_shade_material_(),
  dummy_texture_(),
  //indirect_draw_cmds_(),
  //indirect_draw_buff_(),
  desc_set_layouts_(VK_NULL_HANDLE),
  pipe_layouts_(VK_NULL_HANDLE),
  desc_pool_(VK_NULL_HANDLE),
  desc_sets_(),
  proj_mat_(1.f),
  view_mat_(1.f),
  inv_proj_mat_(1.f),
  inv_view_mat_(1.f),
  cam_(nullptr),
  aniso_sampler_(VK_NULL_HANDLE),
  nearest_sampler_(VK_NULL_HANDLE),
  registered_models_(),
  fullscreenquad_(nullptr),
  current_swapchain_img_(0U) {}

void DeferredRenderer::Init(szt::Camera *cam) {
  cam_ = cam;

  SetupSamplers(vulkan()->device());
  SetupDescriptorPool(vulkan()->device());
  
  model_manager()->set_shade_material_name("g_store");
  model_manager()->set_aniso_sampler(aniso_sampler_);
  model_manager()->set_sets_desc_pool(desc_pool_);
  meshes_heap_manager()->set_aniso_sampler(aniso_sampler_);
  meshes_heap_manager()->set_shade_material_name("vis_store");
  meshes_heap_manager()->set_heap_sets_desc_pool(desc_pool_);

  texture_manager()->Load2DTexture(
     vulkan()->device(),
     kBaseAssetsPath + "dummy.ktx", 
     VK_FORMAT_BC2_UNORM_BLOCK,
     &dummy_texture_,
     aniso_sampler_);

  UpdatePVMatrices();
  SetupMaterials(vulkan()->device());
  SetupRenderPass(vulkan()->device());
  SetupFrameBuffers(vulkan()->device());

}

void DeferredRenderer::Shutdown() {
  vkDeviceWaitIdle(vulkan()->device().device());

  renderpass_.reset(nullptr);
  framebuffers_.clear();

  if (desc_pool_ != VK_NULL_HANDLE) {
    VK_CHECK_RESULT(vkResetDescriptorPool(
        vulkan()->device().device(),
        desc_pool_,
        0U));
    vkDestroyDescriptorPool(
      vulkan()->device().device(),
      desc_pool_,
      nullptr);

    desc_pool_ = VK_NULL_HANDLE;
  }

  if (aniso_sampler_ != VK_NULL_HANDLE) {
    vkDestroySampler(vulkan()->device().device(), aniso_sampler_, nullptr);
    aniso_sampler_ = VK_NULL_HANDLE;
  }
  if (nearest_sampler_ != VK_NULL_HANDLE) {
    vkDestroySampler(vulkan()->device().device(), nearest_sampler_, nullptr);
    nearest_sampler_ = VK_NULL_HANDLE;
  }


  for (uint32_t i = 0U; i < PipeLayoutTypes::num_items; i++) {
    vkDestroyPipelineLayout(
        vulkan()->device().device(),
        pipe_layouts_[i],
        nullptr);
  }
  pipe_layouts_.clear();

  for (uint32_t i = 0U; i < DescSetLayoutTypes::num_items; i++) {
    vkDestroyDescriptorSetLayout(
        vulkan()->device().device(),
        desc_set_layouts_[i],
        nullptr);
  }
  desc_set_layouts_.clear();

  main_static_buff_.Shutdown(vulkan()->device());
}

void DeferredRenderer::PreRender() {
  UpdateBuffers(vulkan()->device());

  vulkan()->swapchain().AcquireNextImage(
      vulkan()->device(),
      vulkan()->image_available_semaphore(),
      current_swapchain_img_);
}

void DeferredRenderer::UpdateBuffers(const VulkanDevice &device) {
  UpdatePVMatrices();
  eastl::vector<Light> transformed_lights;
  UpdateLights(transformed_lights);

  // Cache some sizes
  uint32_t num_mat_instances = material_manager()->GetMaterialInstancesCount();
  uint32_t num_lights = SCAST_U32(transformed_lights.size());
  uint32_t mat4_size = SCAST_U32(sizeof(glm::mat4));
  uint32_t mat4_group_size = mat4_size * 4U;
  uint32_t lights_array_size = (SCAST_U32(sizeof(Light)) * num_lights);
  uint32_t mat_consts_array_size =
    (SCAST_U32(sizeof(MaterialConstants)) * num_mat_instances);

  // Upload data to the buffers
  eastl::array<glm::mat4, 4U> matxs_initial_data = {
    proj_mat_, view_mat_ , inv_proj_mat_, inv_view_mat_};

  void *mapped = nullptr;
  main_static_buff_.Map(device, &mapped);
  uint8_t * mapped_u8 = static_cast<uint8_t *>(mapped);

  memcpy(mapped, matxs_initial_data.data(), mat4_group_size);
  mapped_u8 += mat4_group_size;

  memcpy(mapped_u8, transformed_lights.data(), lights_array_size);
  mapped_u8 += lights_array_size;

  memcpy(mapped_u8, mat_consts_.data(), mat_consts_array_size);

  main_static_buff_.Unmap(device);
}

void DeferredRenderer::Render() {
  VkSemaphore wait_semaphore = vulkan()->image_available_semaphore();
  VkSemaphore signal_semaphore = vulkan()->rendering_finished_semaphore();
  VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
  VkCommandBuffer cmd_buff =
    vulkan()->graphics_queue_cmd_buffers()[current_swapchain_img_];
  VkSubmitInfo submit_info = tools::inits::SubmitInfo();
  submit_info.waitSemaphoreCount = 1U;
  submit_info.pWaitSemaphores = &wait_semaphore;
  submit_info.pWaitDstStageMask = &wait_stage;
  submit_info.commandBufferCount = 1U;
  submit_info.pCommandBuffers = &cmd_buff;
  submit_info.signalSemaphoreCount = 1U;
  submit_info.pSignalSemaphores = &signal_semaphore;

  VK_CHECK_RESULT(vkQueueSubmit(
      vulkan()->device().graphics_queue().queue,
      1U,
      &submit_info,
      VK_NULL_HANDLE));
}

void DeferredRenderer::PostRender() {
  vulkan()->swapchain().Present(
      vulkan()->device().present_queue(),
      vulkan()->rendering_finished_semaphore());
}

void DeferredRenderer::SetupRenderPass(const VulkanDevice &device) {
  renderpass_ = eastl::make_unique<Renderpass>("deferred_full_pass");

  // Colour buffer target 
  uint32_t col_buf_id = renderpass_->AddAttachment(
      0U,
      vulkan()->swapchain().GetSurfaceFormat(),
      VK_SAMPLE_COUNT_1_BIT,
      VK_ATTACHMENT_LOAD_OP_CLEAR,
      VK_ATTACHMENT_STORE_OP_STORE,
      VK_ATTACHMENT_LOAD_OP_DONT_CARE,
      VK_ATTACHMENT_STORE_OP_DONT_CARE,
      VK_IMAGE_LAYOUT_UNDEFINED,
      VK_IMAGE_LAYOUT_PRESENT_SRC_KHR); 

  // Depth buffer target
  uint32_t depth_buf_id = renderpass_->AddAttachment(
      0U,
      device.depth_format(),
      VK_SAMPLE_COUNT_1_BIT,
      VK_ATTACHMENT_LOAD_OP_CLEAR,
      VK_ATTACHMENT_STORE_OP_STORE,
      VK_ATTACHMENT_LOAD_OP_DONT_CARE,
      VK_ATTACHMENT_STORE_OP_DONT_CARE,
      VK_IMAGE_LAYOUT_UNDEFINED,
      VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL); 

  // Maps
  uint32_t diff_albedo_id = renderpass_->AddAttachment(
      0U,
      kDiffuseAlbedoFormat,
      VK_SAMPLE_COUNT_1_BIT,
      VK_ATTACHMENT_LOAD_OP_CLEAR,
      VK_ATTACHMENT_STORE_OP_STORE,
      VK_ATTACHMENT_LOAD_OP_DONT_CARE,
      VK_ATTACHMENT_STORE_OP_DONT_CARE,
      VK_IMAGE_LAYOUT_UNDEFINED,
      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL); 
  uint32_t spec_albedo_id = renderpass_->AddAttachment(
      0U,
      kSpecularAlbedoFormat,
      VK_SAMPLE_COUNT_1_BIT,
      VK_ATTACHMENT_LOAD_OP_CLEAR,
      VK_ATTACHMENT_STORE_OP_STORE,
      VK_ATTACHMENT_LOAD_OP_DONT_CARE,
      VK_ATTACHMENT_STORE_OP_DONT_CARE,
      VK_IMAGE_LAYOUT_UNDEFINED,
      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL); 
  uint32_t norm_id = renderpass_->AddAttachment(
      0U,
      kNormalFormat,
      VK_SAMPLE_COUNT_1_BIT,
      VK_ATTACHMENT_LOAD_OP_CLEAR,
      VK_ATTACHMENT_STORE_OP_STORE,
      VK_ATTACHMENT_LOAD_OP_DONT_CARE,
      VK_ATTACHMENT_STORE_OP_DONT_CARE,
      VK_IMAGE_LAYOUT_UNDEFINED,
      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL); 

  // Accumulation buffer
  uint32_t accum_id = renderpass_->AddAttachment(
      0U,
      kAccumulationFormat,
      VK_SAMPLE_COUNT_1_BIT,
      VK_ATTACHMENT_LOAD_OP_CLEAR,
      VK_ATTACHMENT_STORE_OP_STORE,
      VK_ATTACHMENT_LOAD_OP_DONT_CARE,
      VK_ATTACHMENT_STORE_OP_DONT_CARE,
      VK_IMAGE_LAYOUT_UNDEFINED,
      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL); 



  // Setup first subpass
  uint32_t first_sub_id = renderpass_->AddSubpass(
      "g_store",
      VK_PIPELINE_BIND_POINT_GRAPHICS);
  // G buffers
  renderpass_->AddSubpassColourAttachmentRef(
      first_sub_id,
      diff_albedo_id,
      VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
  renderpass_->AddSubpassColourAttachmentRef(
      first_sub_id,
      spec_albedo_id,
      VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
  renderpass_->AddSubpassColourAttachmentRef(
      first_sub_id,
      norm_id,
      VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
  // Depth
  renderpass_->AddSubpassDepthAttachmentRef(
      first_sub_id,
      depth_buf_id,
      VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

  // Setup second subpass
  uint32_t second_sub_id = renderpass_->AddSubpass(
      "lighting",
      VK_PIPELINE_BIND_POINT_GRAPHICS);
  renderpass_->AddSubpassColourAttachmentRef(
      second_sub_id,
      accum_id,
      VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);

  // Setup third subpass
  uint32_t third_sub_id = renderpass_->AddSubpass(
      "tonemapping",
      VK_PIPELINE_BIND_POINT_GRAPHICS);
  renderpass_->AddSubpassColourAttachmentRef(
      third_sub_id,
      col_buf_id,
      VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);

  // Dependencies
  // Present to colour buffer, which is the last subpass
  renderpass_->AddSubpassDependency(
      VK_SUBPASS_EXTERNAL,
      third_sub_id,
      VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
      VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
      VK_ACCESS_MEMORY_READ_BIT,
      VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
      VK_DEPENDENCY_BY_REGION_BIT);

  // First subpass to second
  renderpass_->AddSubpassDependency(
      first_sub_id,
      second_sub_id,
      VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
      VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
      VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
      VK_ACCESS_SHADER_READ_BIT,
      VK_DEPENDENCY_BY_REGION_BIT);

  // Second subpass to final subpass 
  renderpass_->AddSubpassDependency(
      second_sub_id,
      third_sub_id,
      VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
      VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
      VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
      VK_ACCESS_SHADER_READ_BIT,
      VK_DEPENDENCY_BY_REGION_BIT);

  // Final subpass to present 
  renderpass_->AddSubpassDependency(
      third_sub_id,
      VK_SUBPASS_EXTERNAL,
      VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
      VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
      VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
      VK_ACCESS_MEMORY_READ_BIT,
      VK_DEPENDENCY_BY_REGION_BIT);

  renderpass_->CreateVulkanRenderpass(device);
}

void DeferredRenderer::SetupFrameBuffers(const VulkanDevice &device) {
  // G buffers
  CreateFramebufferAttachment(
      device,
      kDiffuseAlbedoFormat,
      VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
      "diffuse_albedo",
      &g_buffer_[GBtypes::DIFFUSE_ALBEDO]);
  CreateFramebufferAttachment(
      device,
      kSpecularAlbedoFormat,
      VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
      "specular_albedo",
      &g_buffer_[GBtypes::SPECULAR_ALBEDO]);
  CreateFramebufferAttachment(
      device,
      kNormalFormat,
      VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
      "normals",
      &g_buffer_[GBtypes::NORMAL]);

  // Accumulation buffer
  CreateFramebufferAttachment(
      device,
      kAccumulationFormat,
      VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
      "accumulation",
      &accum_buffer_);

  // Depth buffer 
  CreateFramebufferAttachment(
      device,
      device.depth_format(),
      VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
      "depth",
      &depth_buffer_);
  VkImageViewCreateInfo depth_view_create_info =
    tools::inits::ImageViewCreateInfo(
      depth_buffer_->image()->image(),
      VK_IMAGE_VIEW_TYPE_2D,
      depth_buffer_->image()->format(),
      {
        VK_COMPONENT_SWIZZLE_IDENTITY,
        VK_COMPONENT_SWIZZLE_IDENTITY,
        VK_COMPONENT_SWIZZLE_IDENTITY,
        VK_COMPONENT_SWIZZLE_IDENTITY
      },
      {
        VK_IMAGE_ASPECT_DEPTH_BIT,
        0U,
        depth_buffer_->image()->mip_levels(),
        0U,
        1U
      });
  depth_buffer_depth_view_ = depth_buffer_->image()->CreateAdditionalImageView(
      device,
      depth_view_create_info);

  const uint32_t num_swapchain_images = vulkan()->swapchain().GetNumImages();

  for (uint32_t i = 0U; i < num_swapchain_images; i++) {
    eastl::string name;
    name.sprintf("from_swapchain_%d",i);
    eastl::unique_ptr<Framebuffer> frmbuff = eastl::make_unique<Framebuffer>(
        name,
        cam_->viewport().width,
        cam_->viewport().height,
        1U,
        renderpass_.get());

    frmbuff->AddAttachment(vulkan()->swapchain().images()[i]);
    frmbuff->AddAttachment(depth_buffer_);

    for (uint32_t g = 0U; g < GBtypes::num_items; g++) {
      frmbuff->AddAttachment(g_buffer_[g]);
    }

    frmbuff->AddAttachment(accum_buffer_);

    frmbuff->CreateVulkanFramebuffer(device);

    framebuffers_.push_back(eastl::move(frmbuff));
  }
}

void DeferredRenderer::CreateFramebufferAttachment(
    const VulkanDevice &device,
    VkFormat format,
    VkImageUsageFlags img_usage_flags,
    const eastl::string &name,
    VulkanTexture **attachment) const {
  texture_manager()->Create2DTextureFromData(
      device,
      name,
      nullptr,
      0U,
      cam_->viewport().width,
      cam_->viewport().height,
      format,
      attachment,
      VK_NULL_HANDLE,
      img_usage_flags);
}

void DeferredRenderer::RegisterModel(Model &model,
                                     const VertexSetup &g_store_vertex_setup) {
  registered_models_.push_back(&model);

  SetupDescriptorSetAndPipeLayout(vulkan()->device());
  model.CreateAndWriteDescriptorSets(vulkan()->device(),
      desc_set_layouts_[DescSetLayoutTypes::HEAP]);
  SetupUniformBuffers(vulkan()->device());
  SetupMaterialPipelines(vulkan()->device(), g_store_vertex_setup);
  SetupDescriptorSets(vulkan()->device());
  SetupFullscreenQuad(vulkan()->device());
  SetupCommandBuffers(vulkan()->device());
 
  LOG("Registered model in DeferredRenderer.");
}

void DeferredRenderer::SetupMaterials(const VulkanDevice &device) {
  //g_store_material_.Init("g_store");
  //g_shade_material_.Init("g_shade");
  //g_tonemap_material_.Init("g_tone");

  material_manager()->RegisterMaterialName("g_store");
  material_manager()->RegisterMaterialName("g_shade");
  material_manager()->RegisterMaterialName("g_tone");
}

void DeferredRenderer::SetupUniformBuffers(const VulkanDevice &device) {
  // Materials
  mat_consts_ = material_manager()->GetMaterialConstants();
  uint32_t num_mat_instances = material_manager()->GetMaterialInstancesCount();

  // Lights array
  eastl::vector<Light> transformed_lights;
  UpdateLights(transformed_lights);
  uint32_t num_lights = SCAST_U32(transformed_lights.size());

  // Cache some sizes
  uint32_t mat4_size = SCAST_U32(sizeof(glm::mat4));
  uint32_t mat4_group_size = mat4_size * 4U;
  uint32_t lights_array_size = (SCAST_U32(sizeof(Light)) * num_lights);
  uint32_t mat_consts_array_size =
    (SCAST_U32(sizeof(MaterialConstants)) * num_mat_instances);

  // Main static buffer
  VulkanBufferInitInfo buff_init_info;
  buff_init_info.size = mat4_group_size +
    lights_array_size +
    mat_consts_array_size;
  buff_init_info.memory_property_flags = /*VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT |*/
    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
  buff_init_info.buffer_usage_flags = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
  main_static_buff_.Init(device, buff_init_info);

  // Upload data to it
  eastl::array<glm::mat4, 4U> matxs_initial_data = {
    proj_mat_, view_mat_ , inv_proj_mat_, inv_view_mat_};

  void *mapped = nullptr;
  main_static_buff_.Map(device, &mapped);
  uint8_t * mapped_u8 = static_cast<uint8_t *>(mapped);

  memcpy(mapped, matxs_initial_data.data(), mat4_group_size);
  mapped_u8 += mat4_group_size;

  memcpy(mapped_u8, transformed_lights.data(), lights_array_size);
  mapped_u8 += lights_array_size;

  memcpy(mapped_u8, mat_consts_.data(), mat_consts_array_size);

  main_static_buff_.Unmap(device);
}

void DeferredRenderer::SetupDescriptorPool(const VulkanDevice &device) {
  std::vector<VkDescriptorPoolSize> pool_sizes;

  // Uniforms
  pool_sizes.push_back(tools::inits::DescriptorPoolSize(
      VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
      kMaxNumUniformBuffers));

  // Framebuffers
  pool_sizes.push_back(tools::inits::DescriptorPoolSize(
      VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
      kMaxNumMatInstances * 
        SCAST_U32(MatTextureType::size) + 10U));

  // Storage buffers
  pool_sizes.push_back(tools::inits::DescriptorPoolSize(
      VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      kMaxNumSSBOs));

  VkDescriptorPoolCreateInfo pool_create_info =
    tools::inits::DescriptrorPoolCreateInfo(
      DescSetLayoutTypes::num_items,
      SCAST_U32(pool_sizes.size()),
      pool_sizes.data());

  VK_CHECK_RESULT(vkCreateDescriptorPool(device.device(), &pool_create_info,
                  nullptr, &desc_pool_));
}

void DeferredRenderer::SetupDescriptorSetAndPipeLayout(
    const VulkanDevice &device) {
  eastl::vector<std::vector<VkDescriptorSetLayoutBinding>> bindings(
      DescSetLayoutTypes::num_items);

  // Main static buffer
  bindings[DescSetLayoutTypes::GPASS_GENERIC].push_back(
    tools::inits::DescriptorSetLayoutBinding(
      kMainStaticBuffBindingPos,
      VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      1U,
      VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
      nullptr));
  
  // Lights array
  bindings[DescSetLayoutTypes::GPASS_GENERIC].push_back(
    tools::inits::DescriptorSetLayoutBinding(
      kLightsArrayBindingPos,
      VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      1U,
      VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
      nullptr));
  
  // Material constants array
  bindings[DescSetLayoutTypes::GPASS_GENERIC].push_back(
    tools::inits::DescriptorSetLayoutBinding(
      kMatConstsArrayBindingPos,
      VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      1U,
      VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
      nullptr));

  // Model matrices for all meshes
  bindings[DescSetLayoutTypes::HEAP].push_back(
    tools::inits::DescriptorSetLayoutBinding(
      kModelMatxsBufferBindPos,
      VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      1U,
      VK_SHADER_STAGE_VERTEX_BIT |
        VK_SHADER_STAGE_FRAGMENT_BIT,
      nullptr));
  
  // Vertex buffer
  for (uint32_t i = 0U; i < SCAST_U32(VertexElementType::num_items); ++i) {
    bindings[DescSetLayoutTypes::HEAP].push_back(
      tools::inits::DescriptorSetLayoutBinding(
        kVertexBuffersBaseBindPos + i,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        1U,
        VK_SHADER_STAGE_FRAGMENT_BIT,
        nullptr));
  }

  // Index buffer
  bindings[DescSetLayoutTypes::HEAP].push_back(
    tools::inits::DescriptorSetLayoutBinding(
      kIdxBufferBindPos,
      VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      1U,
      VK_SHADER_STAGE_FRAGMENT_BIT,
      nullptr));


  // Indirect draw buffers
  bindings[DescSetLayoutTypes::HEAP].push_back(
    tools::inits::DescriptorSetLayoutBinding(
      kIndirectDrawCmdsBindingPos,
      VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      1U,
      VK_SHADER_STAGE_FRAGMENT_BIT,
      nullptr));

  // Material IDs
  bindings[DescSetLayoutTypes::HEAP].push_back(
    tools::inits::DescriptorSetLayoutBinding(
      kMaterialIDsBufferBindPos,
      VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      1U,
      VK_SHADER_STAGE_FRAGMENT_BIT,
      nullptr));

  // Depth buffer
  bindings[DescSetLayoutTypes::GPASS_GENERIC].push_back(
    tools::inits::DescriptorSetLayoutBinding(
      kDepthBuffBindingPos,
      VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
      1U,
      VK_SHADER_STAGE_FRAGMENT_BIT,
      nullptr));

  uint32_t num_mat_instances = material_manager()->GetMaterialInstancesCount();
  // Diffuse textures as combined image samplers
  bindings[DescSetLayoutTypes::GPASS_GENERIC].push_back(
    tools::inits::DescriptorSetLayoutBinding(
      kDiffuseTexturesArrayBindingPos,
      VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
      num_mat_instances,
      VK_SHADER_STAGE_FRAGMENT_BIT,
      nullptr));
  // Ambient textures as combined image samplers 
  bindings[DescSetLayoutTypes::GPASS_GENERIC].push_back(
    tools::inits::DescriptorSetLayoutBinding(
      kAmbientTexturesArrayBindingPos,
      VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
      num_mat_instances,
      VK_SHADER_STAGE_FRAGMENT_BIT,
      nullptr));
  // Specular textures as combined image samplers 
  bindings[DescSetLayoutTypes::GPASS_GENERIC].push_back(
    tools::inits::DescriptorSetLayoutBinding(
      kSpecularTexturesArrayBindingPos,
      VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
      num_mat_instances,
      VK_SHADER_STAGE_FRAGMENT_BIT,
      nullptr));
  // Normal textures as combined image samplers 
  bindings[DescSetLayoutTypes::GPASS_GENERIC].push_back(
    tools::inits::DescriptorSetLayoutBinding(
      kNormalTexturesArrayBindingPos,
      VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
      num_mat_instances,
      VK_SHADER_STAGE_FRAGMENT_BIT,
      nullptr));
  // Roughness textures as combined image samplers 
  bindings[DescSetLayoutTypes::GPASS_GENERIC].push_back(
    tools::inits::DescriptorSetLayoutBinding(
      kRoughnessTexturesArrayBindingPos,
      VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
      num_mat_instances,
      VK_SHADER_STAGE_FRAGMENT_BIT,
      nullptr));

  // Accumulation buffer
  bindings[DescSetLayoutTypes::GPASS_GENERIC].push_back(
    tools::inits::DescriptorSetLayoutBinding(
      kAccumulationBufferBindingPos,
      VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
      1U,
      VK_SHADER_STAGE_FRAGMENT_BIT,
      nullptr));

  // G-Buffers
  for (uint32_t i = 0U; i < GBtypes::num_items; i++) {
    bindings[DescSetLayoutTypes::GPASS_GENERIC].push_back(
      tools::inits::DescriptorSetLayoutBinding(
        kGBufferBaseBindingPos + i,
        VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        1U,
        VK_SHADER_STAGE_FRAGMENT_BIT,
        nullptr));
  }

  desc_set_layouts_.resize(DescSetLayoutTypes::num_items);

  for (uint32_t i = 0U; i < DescSetLayoutTypes::num_items; i++) {
    VkDescriptorSetLayoutCreateInfo set_layout_create_info =
      tools::inits::DescriptrorSetLayoutCreateInfo();
    set_layout_create_info.bindingCount =
      SCAST_U32(bindings[i].size());
    set_layout_create_info.pBindings =
      bindings[i].data();

    VK_CHECK_RESULT(vkCreateDescriptorSetLayout(
        device.device(),
        &set_layout_create_info,
        nullptr,
        &desc_set_layouts_[i]));

    LOG("Desc set layout: " << desc_set_layouts_[i] << " b count: " << 
        set_layout_create_info.bindingCount);
  }

  eastl::array<VkDescriptorSetLayout, SetTypes::num_items>
  local_layouts = {
    desc_set_layouts_[DescSetLayoutTypes::GPASS_GENERIC],
  };
  VkDescriptorSetAllocateInfo set_allocate_info =
    tools::inits::DescriptorSetAllocateInfo(
      desc_pool_,
      SCAST_U32(local_layouts.size()),
      local_layouts.data());

  VK_CHECK_RESULT(vkAllocateDescriptorSets(
        device.device(),
        &set_allocate_info,
        desc_sets_.data()));

  // Create pipeline layouts
  pipe_layouts_.resize(PipeLayoutTypes::num_items);

  // Push constant for the meshes ID
  VkPushConstantRange push_const_range = {
    VK_SHADER_STAGE_VERTEX_BIT,
    0U,
    SCAST_U32(sizeof(uint32_t))
  };

  VkPipelineLayoutCreateInfo pipe_layout_create_info =     tools::inits::PipelineLayoutCreateInfo(
      DescSetLayoutTypes::num_items, // Desc set layouts up to VISBUFF
      desc_set_layouts_.data(),
      1U,
      &push_const_range);

  VK_CHECK_RESULT(vkCreatePipelineLayout(
      device.device(),
      &pipe_layout_create_info,
      nullptr,
      &pipe_layouts_[PipeLayoutTypes::GPASS]));
}

void DeferredRenderer::SetupDescriptorSets(const VulkanDevice &device) {
  // Update the descriptor set
  eastl::vector<VkWriteDescriptorSet> write_desc_sets;

  // Cache some sizes
  uint32_t num_mat_instances = material_manager()->GetMaterialInstancesCount();
  uint32_t num_lights = lights_manager()->GetNumLights();
  uint32_t mat4_size = SCAST_U32(sizeof(glm::mat4));
  uint32_t mat4_group_size = mat4_size * 4U;
  uint32_t lights_array_size = (SCAST_U32(sizeof(Light)) * num_lights);
  uint32_t mat_consts_array_size =
    (SCAST_U32(sizeof(MaterialConstants)) * num_mat_instances);

  // Main static buffer
  VkDescriptorBufferInfo desc_main_static_buff_info =
    main_static_buff_.GetDescriptorBufferInfo(mat4_group_size);
  write_desc_sets.push_back(tools::inits::WriteDescriptorSet(
      desc_sets_[SetTypes::GPASS_GENERIC],
      kMainStaticBuffBindingPos,
      0U,
      1U,
      VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      nullptr,
      &desc_main_static_buff_info,
      nullptr));
  
  // Lights array
  VkDescriptorBufferInfo desc_lights_array_info =
    main_static_buff_.GetDescriptorBufferInfo(lights_array_size, mat4_group_size);
  write_desc_sets.push_back(tools::inits::WriteDescriptorSet(
      desc_sets_[SetTypes::GPASS_GENERIC],
      kLightsArrayBindingPos,
      0U,
      1U,
      VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      nullptr,
      &desc_lights_array_info,
      nullptr));
  
  // Material constants array
  VkDescriptorBufferInfo desc_mat_consts_info =
    main_static_buff_.GetDescriptorBufferInfo(mat_consts_array_size,
                                              mat4_group_size + lights_array_size);
  write_desc_sets.push_back(tools::inits::WriteDescriptorSet(
      desc_sets_[SetTypes::GPASS_GENERIC],
      kMatConstsArrayBindingPos,
      0U,
      1U,
      VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      nullptr,
      &desc_mat_consts_info,
      nullptr));

  // Depth buffer
  VkDescriptorImageInfo depth_buff_img_info =
    depth_buffer_->image()->GetDescriptorImageInfo(nearest_sampler_);
  depth_buff_img_info.imageView = *depth_buffer_depth_view_;
  write_desc_sets.push_back(tools::inits::WriteDescriptorSet(
      desc_sets_[SetTypes::GPASS_GENERIC],
      kDepthBuffBindingPos,
      0U,
      1U,
      VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
      &depth_buff_img_info,
      nullptr,
      nullptr));

  eastl::vector<VkDescriptorImageInfo> diff_descs_image_infos;
  material_manager()->GetDescriptorImageInfosByType(
      MatTextureType::DIFFUSE,
      diff_descs_image_infos);

  // Diffuse textures as combined image samplers
  write_desc_sets.push_back(tools::inits::WriteDescriptorSet(
      desc_sets_[SetTypes::GPASS_GENERIC],
      kDiffuseTexturesArrayBindingPos,
      0U,
      SCAST_U32(diff_descs_image_infos.size()),
      VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
      diff_descs_image_infos.data(),
      nullptr,
      nullptr));
  
  eastl::vector<VkDescriptorImageInfo> amb_descs_image_infos;
  material_manager()->GetDescriptorImageInfosByType(
      MatTextureType::AMBIENT,
      amb_descs_image_infos);

  // Ambient textures as combined image samplers
  write_desc_sets.push_back(tools::inits::WriteDescriptorSet(
      desc_sets_[SetTypes::GPASS_GENERIC],
      kAmbientTexturesArrayBindingPos,
      0U,
      SCAST_U32(amb_descs_image_infos.size()),
      VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
      amb_descs_image_infos.data(),
      nullptr,
      nullptr));
  
  eastl::vector<VkDescriptorImageInfo> spec_descs_image_infos;
  material_manager()->GetDescriptorImageInfosByType(
      MatTextureType::SPECULAR,
      spec_descs_image_infos);

  // Specular textures as combined image samplers
  write_desc_sets.push_back(tools::inits::WriteDescriptorSet(
      desc_sets_[SetTypes::GPASS_GENERIC],
      kSpecularTexturesArrayBindingPos,
      0U,
      SCAST_U32(spec_descs_image_infos.size()),
      VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
      spec_descs_image_infos.data(),
      nullptr,
      nullptr));
  
  eastl::vector<VkDescriptorImageInfo> rough_descs_image_infos;
  material_manager()->GetDescriptorImageInfosByType(
      MatTextureType::SPECULAR_HIGHLIGHT,
      rough_descs_image_infos);

  // Roughness textures as combined image samplers
  write_desc_sets.push_back(tools::inits::WriteDescriptorSet(
      desc_sets_[SetTypes::GPASS_GENERIC],
      kRoughnessTexturesArrayBindingPos,
      0U,
      SCAST_U32(rough_descs_image_infos.size()),
      VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
      rough_descs_image_infos.data(),
      nullptr,
      nullptr));
  
  eastl::vector<VkDescriptorImageInfo> norm_descs_image_infos;
  material_manager()->GetDescriptorImageInfosByType(
      MatTextureType::NORMAL,
      norm_descs_image_infos);

  // Roughness textures as combined image samplers
  write_desc_sets.push_back(tools::inits::WriteDescriptorSet(
      desc_sets_[SetTypes::GPASS_GENERIC],
      kNormalTexturesArrayBindingPos,
      0U,
      SCAST_U32(norm_descs_image_infos.size()),
      VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
      norm_descs_image_infos.data(),
      nullptr,
      nullptr));


  // Accumulation buffer
  VkDescriptorImageInfo accum_buff_img_info =
    accum_buffer_->image()->GetDescriptorImageInfo(nearest_sampler_);
  write_desc_sets.push_back(tools::inits::WriteDescriptorSet(
      desc_sets_[SetTypes::GPASS_GENERIC],
      kAccumulationBufferBindingPos,
      0U,
      1U,
      VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
      &accum_buff_img_info,
      nullptr,
      nullptr));

  
  // G Buffer
  eastl::array<VkDescriptorImageInfo, GBtypes::num_items> g_buff_img_infos;
  for (uint32_t i = 0U; i < GBtypes::num_items; i++) {
    g_buff_img_infos[i] =
      g_buffer_[i]->image()->GetDescriptorImageInfo(nearest_sampler_);
    write_desc_sets.push_back(tools::inits::WriteDescriptorSet(
        desc_sets_[SetTypes::GPASS_GENERIC],
        kGBufferBaseBindingPos + i,
        0U,
        1U,
        VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        &g_buff_img_infos[i],
        nullptr,
        nullptr));
  } 

  // Update them 
  vkUpdateDescriptorSets(
      device.device(),
      SCAST_U32(write_desc_sets.size()),
      write_desc_sets.data(),
      0U,
      nullptr);
}

void DeferredRenderer::UpdatePVMatrices() {
  proj_mat_ = cam_->projection_mat();
  view_mat_ = cam_->view_mat();
  inv_proj_mat_ = glm::inverse(proj_mat_);
  inv_view_mat_ = glm::inverse(view_mat_);
}

void DeferredRenderer::SetupCommandBuffers(const VulkanDevice &device) {
  // Cache common settings to all command buffers 
  VkCommandBufferBeginInfo cmd_buff_begin_info =
    tools::inits::CommandBufferBeginInfo(
        VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT);
  cmd_buff_begin_info.pInheritanceInfo = nullptr;

  std::vector<VkClearValue> clear_values;
  VkClearValue clear_value;
  clear_value.color = {{0.f, 0.f, 0.f, 0.f}};
  clear_values.push_back(clear_value);
  clear_value.depthStencil = {1.f, 0U};
  clear_values.push_back(clear_value);
  clear_value.color = {{0.f, 0.f, 0.f, 0.f}};
  clear_values.push_back(clear_value);
  clear_value.color = {{0.f, 0.f, 0.f, 0.f}};
  clear_values.push_back(clear_value);
  clear_value.color = {{0.f, 0.f, 0.f, 0.f}};
  clear_values.push_back(clear_value);
  clear_value.color = {{0.f, 0.f, 0.f, 0.f}};
  clear_values.push_back(clear_value);

  // Record command buffers
  const std::vector<VkCommandBuffer> &graphics_buffs =
    vulkan()->graphics_queue_cmd_buffers();
  uint32_t num_swapchain_images = vulkan()->swapchain().GetNumImages();
  for (uint32_t i = 0U; i < num_swapchain_images; i++) {
    VK_CHECK_RESULT(vkBeginCommandBuffer(
        graphics_buffs[i], &cmd_buff_begin_info));

    renderpass_->BeginRenderpass(
        graphics_buffs[i],
        VK_SUBPASS_CONTENTS_INLINE,
        framebuffers_[i].get(),
        {0U, 0U, cam_->viewport().width, cam_->viewport().height},
        SCAST_U32(clear_values.size()),
        clear_values.data());

    g_store_material_->BindPipeline(graphics_buffs[i],
                                    VK_PIPELINE_BIND_POINT_GRAPHICS);

    vkCmdBindDescriptorSets(
        graphics_buffs[i],
        VK_PIPELINE_BIND_POINT_GRAPHICS,
        pipe_layouts_[PipeLayoutTypes::GPASS],
        0U,
        DescSetLayoutTypes::HEAP,
        desc_sets_.data(),  
        0U,
        nullptr);

    for (eastl::vector<Model*>::iterator itor =
           registered_models_.begin();
         itor != registered_models_.end();
         ++itor) {
      (*itor)->BindVertexBuffer(graphics_buffs[i]);
      (*itor)->BindIndexBuffer(graphics_buffs[i]);
      (*itor)->RenderMeshesByMaterial(
          graphics_buffs[i],
          pipe_layouts_[PipeLayoutTypes::GPASS],
          DescSetLayoutTypes::HEAP);
    }

    // Light shading pass
    renderpass_->NextSubpass(graphics_buffs[i], VK_SUBPASS_CONTENTS_INLINE);

    g_shade_material_->BindPipeline(graphics_buffs[i],
                                    VK_PIPELINE_BIND_POINT_GRAPHICS);

    fullscreenquad_->BindVertexBuffer(graphics_buffs[i]);  
    fullscreenquad_->BindIndexBuffer(graphics_buffs[i]);  

    vkCmdDrawIndexed(
        graphics_buffs[i],
        6U,
        1U,
        0U,
        0U,
        0U);

    // Light shading pass
    renderpass_->NextSubpass(graphics_buffs[i], VK_SUBPASS_CONTENTS_INLINE);

    g_tonemap_material_->BindPipeline(graphics_buffs[i],
                                      VK_PIPELINE_BIND_POINT_GRAPHICS);

    vkCmdDrawIndexed(
        graphics_buffs[i],
        6U,
        1U,
        0U,
        0U,
        0U);

    //vkCmdEndRenderPass(graphics_buffs[i]);
    renderpass_->EndRenderpass(graphics_buffs[i]);

    VK_CHECK_RESULT(vkEndCommandBuffer(graphics_buffs[i]));
  }
}

void DeferredRenderer::SetupSamplers(const VulkanDevice &device) {
  // Create an aniso sampler
  VkSamplerCreateInfo sampler_create_info = tools::inits::SamplerCreateInfo(
      VK_FILTER_LINEAR,
      VK_FILTER_LINEAR,
      VK_SAMPLER_MIPMAP_MODE_LINEAR,
      VK_SAMPLER_ADDRESS_MODE_REPEAT,
      VK_SAMPLER_ADDRESS_MODE_REPEAT,
      VK_SAMPLER_ADDRESS_MODE_REPEAT,
      0.f,
      VK_TRUE,
      device.physical_properties().limits.maxSamplerAnisotropy,
      VK_FALSE,
      VK_COMPARE_OP_NEVER,
      0.f,
      1.f,
      VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE,
      VK_FALSE);
  
  VK_CHECK_RESULT(vkCreateSampler(
      device.device(),
      &sampler_create_info,
      nullptr,
      &aniso_sampler_));

  // Create a nearest neighbour sampler
  sampler_create_info = tools::inits::SamplerCreateInfo(
      VK_FILTER_NEAREST,
      VK_FILTER_NEAREST,
      VK_SAMPLER_MIPMAP_MODE_NEAREST,
      VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
      VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
      VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
      0.f,
      VK_FALSE,
      0U,
      VK_FALSE,
      VK_COMPARE_OP_NEVER,
      0.f,
      1.f,
      VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE,
      VK_FALSE);

  VK_CHECK_RESULT(vkCreateSampler(
      device.device(),
      &sampler_create_info,
      nullptr,
      &nearest_sampler_));
}
  
void DeferredRenderer::SetupMaterialPipelines(
    const VulkanDevice &device,
    const VertexSetup &g_store_vertex_setup) {
  eastl::vector<VertexElement> vtx_layout;
  vtx_layout.push_back(VertexElement(
        VertexElementType::POSITION,
        SCAST_U32(sizeof(glm::vec3)),
        VK_FORMAT_R32G32B32_SFLOAT));

  VertexSetup vertex_setup_quads(vtx_layout);

  // Setup visibility shade material
  eastl::unique_ptr<MaterialShader> g_shade_frag =
    eastl::make_unique<MaterialShader>(
      kBaseShaderAssetsPath + "g_shade.frag",
      "main",
      ShaderTypes::FRAGMENT);
  
  eastl::unique_ptr<MaterialShader> g_shade_vert =
    eastl::make_unique<MaterialShader>(
      kBaseShaderAssetsPath + "g_shade.vert",
      "main",
      ShaderTypes::VERTEX);
  
  uint32_t num_materials = material_manager()->GetMaterialInstancesCount();
  g_shade_frag->AddSpecialisationEntry(
      kNumMaterialsSpecConstPos,
      SCAST_U32(sizeof(uint32_t)),
      &num_materials);
  uint32_t num_lights = lights_manager()->GetNumLights();
  g_shade_frag->AddSpecialisationEntry(
      kNumLightsSpecConstPos,
      SCAST_U32(sizeof(uint32_t)),
      &num_lights);
  g_shade_vert->AddSpecialisationEntry(
      kNumMaterialsSpecConstPos,
      SCAST_U32(sizeof(uint32_t)),
      &num_materials);
  g_shade_vert->AddSpecialisationEntry(
      kNumLightsSpecConstPos,
      SCAST_U32(sizeof(uint32_t)),
      &num_lights);
  
  eastl::unique_ptr<MaterialBuilder> builder_shade =
    eastl::make_unique<MaterialBuilder>(
    vertex_setup_quads,
    "g_shade",
    pipe_layouts_[PipeLayoutTypes::GPASS],
    renderpass_->GetVkRenderpass(),
    VK_FRONT_FACE_CLOCKWISE,
    1U,
    cam_->viewport());

  builder_shade->AddColorBlendAttachment(
      VK_FALSE,
      VK_BLEND_FACTOR_ONE,
      VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
      VK_BLEND_OP_ADD,
      VK_BLEND_FACTOR_ONE,
      VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
      VK_BLEND_OP_ADD,
      0xf);
  float blend_constants[4U] = { 1.f, 1.f, 1.f, 1.f };
  builder_shade->AddColorBlendStateCreateInfo(
      VK_FALSE,
      VK_LOGIC_OP_SET,
      blend_constants);
  builder_shade->AddShader(eastl::move(g_shade_vert));
  builder_shade->AddShader(eastl::move(g_shade_frag));

  g_shade_material_ =
    material_manager()->CreateMaterial(device, eastl::move(builder_shade)); 

  // Setup store material
  eastl::unique_ptr<MaterialShader> g_store_frag =
    eastl::make_unique<MaterialShader>(
      kBaseShaderAssetsPath + "g_store.frag",
      "main",
      ShaderTypes::FRAGMENT);
  
  eastl::unique_ptr<MaterialShader> g_store_vert =
    eastl::make_unique<MaterialShader>(
      kBaseShaderAssetsPath + "g_store.vert",
      "main",
      ShaderTypes::VERTEX);
  
  g_store_vert->AddSpecialisationEntry(
      kNumMaterialsSpecConstPos,
      SCAST_U32(sizeof(uint32_t)),
      &num_materials);
  g_store_vert->AddSpecialisationEntry(
      kNumLightsSpecConstPos,
      SCAST_U32(sizeof(uint32_t)),
      &num_lights);

  eastl::unique_ptr<MaterialBuilder> builder_store =
    eastl::make_unique<MaterialBuilder>(
    g_store_vertex_setup,
    "g_store",
    pipe_layouts_[PipeLayoutTypes::GPASS],
    renderpass_->GetVkRenderpass(),
    VK_FRONT_FACE_COUNTER_CLOCKWISE,
    0U,
    cam_->viewport());
  
  for (uint32_t i = 0U; i < GBtypes::num_items; i++) {
    builder_store->AddColorBlendAttachment(
        VK_FALSE,
        VK_BLEND_FACTOR_ONE,
        VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
        VK_BLEND_OP_ADD,
        VK_BLEND_FACTOR_ONE,
        VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
        VK_BLEND_OP_ADD,
        0xf);
  }
  builder_store->AddColorBlendStateCreateInfo(
      VK_FALSE,
      VK_LOGIC_OP_SET,
      blend_constants);
  builder_store->AddShader(eastl::move(g_store_vert));
  builder_store->AddShader(eastl::move(g_store_frag));
  builder_store->SetDepthTestEnable(VK_TRUE);
  builder_store->SetDepthWriteEnable(VK_TRUE);

  g_store_material_ =
    material_manager()->CreateMaterial(device, eastl::move(builder_store)); 

  // Setup tonemap material
  eastl::unique_ptr<MaterialShader> tone_frag =
    eastl::make_unique<MaterialShader>(
      kBaseShaderAssetsPath + "tonemapping.frag",
      "main",
      ShaderTypes::FRAGMENT);

  eastl::unique_ptr<MaterialShader> tone_vert =
    eastl::make_unique<MaterialShader>(
      kBaseShaderAssetsPath + "tonemapping.vert",
      "main",
      ShaderTypes::VERTEX);

  eastl::unique_ptr<MaterialBuilder> builder_tone =
    eastl::make_unique<MaterialBuilder>(
    vertex_setup_quads,
    "g_tone",
    pipe_layouts_[PipeLayoutTypes::GPASS],
    renderpass_->GetVkRenderpass(),
    VK_FRONT_FACE_CLOCKWISE,
    2U,
    cam_->viewport());

  builder_tone->AddColorBlendAttachment(
      VK_FALSE,
      VK_BLEND_FACTOR_ONE,
      VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
      VK_BLEND_OP_ADD,
      VK_BLEND_FACTOR_ONE,
      VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
      VK_BLEND_OP_ADD,
      0xf);
  builder_tone->AddColorBlendStateCreateInfo(
      VK_FALSE,
      VK_LOGIC_OP_SET,
      blend_constants);
  builder_tone->AddShader(eastl::move(tone_vert));
  builder_tone->AddShader(eastl::move(tone_frag));

  g_tonemap_material_ =
    material_manager()->CreateMaterial(device, eastl::move(builder_tone)); 
}

void DeferredRenderer::SetupFullscreenQuad(const VulkanDevice &device){
  eastl::vector<VertexElement> vtx_layout;
  vtx_layout.push_back(VertexElement(
        VertexElementType::POSITION,
        SCAST_U32(sizeof(glm::vec3)),
        VK_FORMAT_R32G32B32_SFLOAT));

  VertexSetup vertex_setup_quads(vtx_layout);

  // Group all vertex data together
  ModelBuilder model_builder(
    vertex_setup_quads,
    desc_pool_);

  Vertex vtx;
  vtx.pos = { -1.f, 1.f, 0.f };
  model_builder.AddVertex(vtx);
  vtx.pos = { -1.f, -1.f, 0.f };
  model_builder.AddVertex(vtx);
  vtx.pos = { 1.f, -1.f, 0.f };
  model_builder.AddVertex(vtx);
  vtx.pos = { 1.f, 1.f, 0.f };
  model_builder.AddVertex(vtx);

  model_builder.AddIndex(0U);
  model_builder.AddIndex(1U);
  model_builder.AddIndex(2U);
  model_builder.AddIndex(0U);
  model_builder.AddIndex(2U);
  model_builder.AddIndex(3U);

  Mesh quad_mesh(
    0U,
    6U,
    0U,
    0U);

  model_builder.AddMesh(&quad_mesh);

  model_manager()->CreateModel(device, "fullscreenquad", model_builder,
                               &fullscreenquad_);
}

void DeferredRenderer::UpdateLights(eastl::vector<Light> &transformed_lights) {
  transformed_lights = lights_manager()->TransformLights(view_mat_);
}

void DeferredRenderer::ReloadAllShaders() {
  material_manager()->ReloadAllShaders(vulkan()->device());

  SetupCommandBuffers(vulkan()->device());
}

} // namespace vks
