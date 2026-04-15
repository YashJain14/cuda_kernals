with open("stage11_include/static_kernel_configuration.cuh", "r") as f:
    content = f.read()

content = content.replace("template <FlashForwardKernelConfig CFG>", "template <const FlashForwardKernelConfig &CFG>")
content = content.replace("template <FlashForwardKernelConfig _CFG>", "template <const FlashForwardKernelConfig &_CFG>")

with open("stage11_include/static_kernel_configuration.cuh", "w") as f:
    f.write(content)
