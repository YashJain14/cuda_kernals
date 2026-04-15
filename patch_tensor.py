with open("stage11_include/tensor.cuh", "r") as f:
    content = f.read()

content = content.replace("using GSMemShape = gsmem::TensorShape;", "using GSMemShape = typename gsmem::TensorShape;")
content = content.replace("using SmemStride = gsmem::SmemStride;", "using SmemStride = typename gsmem::SmemStride;")
content = content.replace("using GMemStride = gsmem::GMemStride;", "using GMemStride = typename gsmem::GMemStride;")

with open("stage11_include/tensor.cuh", "w") as f:
    f.write(content)
