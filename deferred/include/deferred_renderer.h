#ifndef VKS_DEFERREDRENDERER
#define VKS_DEFERREDRENDERER

#include <vulkan/vulkan.h>
#include <vulkan_image.h>
#include <material.h>
#include <vulkan_buffer.h>
#include <glm/glm.hpp>
#include <EASTL/array.h>
#include <EASTL/vector.h>
#include <EASTL/string.h>
#include <EASTL/unique_ptr.h>
#include <light.h>
#include <renderpass.h>
#include <framebuffer.h>

namespace szt {
  class Camera; 
} // namespace szt

namespace vks {

class VulkanDevice;
class Model;
class VulkanTexture;
class VertexSetup;
class ModelWithHeaps;

struct SetsEnum {
  enum Sets {
    GPASS_GENERIC = 0U,
    num_items
  }; // enum Sets 
}; // struct SetsEnum
typedef SetsEnum::Sets SetTypes;

struct DescSetLayoutsEnum {
  enum DescSetLayouts {
    GPASS_GENERIC = 0U,
    HEAP,
    num_items
  }; // enum DescSetLayouts
}; // struct DescSetLayoutsEnum
typedef DescSetLayoutsEnum::DescSetLayouts DescSetLayoutTypes;

class DeferredRenderer {
 public:
  DeferredRenderer();

  void Init(szt::Camera *cam);

  void Shutdown();
  void PreRender();
  void Render();
  void PostRender();

  void ReloadAllShaders();

  // Register a model for rendering.
  // - Create necessary indirect draw calls and update relative buffer
  void RegisterModel(Model &model,
                     const VertexSetup &g_store_vertex_setup);
  

 private:
  void SetupRenderPass(const VulkanDevice &device);
  void SetupFrameBuffers(const VulkanDevice &device);
  void SetupMaterials(const VulkanDevice &device);
  void SetupMaterialPipelines(const VulkanDevice &device,
                              const VertexSetup &g_store_vertex_setup);
  void SetupUniformBuffers(const VulkanDevice &device);
  // Create both the desc set layouts and the pipe layouts
  void SetupDescriptorSetAndPipeLayout(const VulkanDevice &device);
  void SetupDescriptorSets(const VulkanDevice &device);
  void SetupDescriptorPool(const VulkanDevice &device);
  void SetupCommandBuffers(const VulkanDevice &device);
  void SetupSamplers(const VulkanDevice &device);
  void UpdatePVMatrices();
  void UpdateBuffers(const VulkanDevice &device);
  void UpdateLights(eastl::vector<Light> &transformed_lights);
  void SetupFullscreenQuad(const VulkanDevice &device);
  void GenerateSSAOKernel();
  void GenerateNoiseTextureData();
  void CreateFramebufferAttachment(
      const VulkanDevice &device,
      VkFormat format,
      VkImageUsageFlags img_usage_flags,
      const eastl::string &name,
      VulkanTexture **attachment) const;
      //VkCommandBuffer cmd_buff);

  eastl::unique_ptr<Renderpass> renderpass_;

  /**
   * @brief Havee as many framebuffs as there are swapchain images
   */
  eastl::vector<eastl::unique_ptr<Framebuffer>> framebuffers_;
  uint32_t current_swapchain_img_;

  /**
   * @brief Have as many command buffers as there are swapchain images
   */
  eastl::vector<VkCommandBuffer> cmd_buffers_;

  struct GBuffersEnum {
    enum GBuffers {
      DIFFUSE_ALBEDO = 0U,
      SPECULAR_ALBEDO,
      NORMAL,
      num_items
    }; // enum GBuffers
  }; // struct GBuffersEnum
  typedef GBuffersEnum::GBuffers GBtypes;
  eastl::array<VulkanTexture *, GBtypes::num_items> g_buffer_;
  VulkanTexture *accum_buffer_;
  VulkanTexture *depth_buffer_;
  VkImageView *depth_buffer_depth_view_;

  Material *g_store_material_;
  Material *g_shade_material_;
  Material *g_tonemap_material_;

  /**
   * @brief Texture used in replacement in materials which don't have a texture
   *        for a given type of map.
   */
  VulkanTexture *dummy_texture_;

  // Used to render all geometry in one API call
  //std::vector<VkDrawIndexedIndirectCommand> indirect_draw_cmds_;
  //VulkanBuffer indirect_draw_buff_;

  eastl::vector<VkDescriptorSetLayout> desc_set_layouts_;
  eastl::array<VkDescriptorSet, SetTypes::num_items> desc_sets_;
  VkDescriptorPool desc_pool_;

  struct PipeLayoutsEnum {
    enum PipeLayouts {
      GPASS = 0U,
      num_items
    }; // enum PipeLayouts
  }; // struct PipeLayoutsEnum
  typedef PipeLayoutsEnum::PipeLayouts PipeLayoutTypes;
  eastl::vector<VkPipelineLayout> pipe_layouts_;

  VulkanBuffer main_static_buff_;
  
  // These are contained in camera, but this way they can be easily used to
  // update the VulkanBuffers
  glm::mat4 proj_mat_;
  glm::mat4 view_mat_;
  glm::mat4 inv_proj_mat_;
  glm::mat4 inv_view_mat_;

  szt::Camera *cam_;

  VkSampler aniso_sampler_;
  VkSampler nearest_sampler_;

  eastl::vector<Model*> registered_models_;
  Model *fullscreenquad_;

  eastl::vector<MaterialConstants> mat_consts_;
}; // class DeferredRenderer

} // namespace vks

#endif
