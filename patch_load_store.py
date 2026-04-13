with open("stage12_include/load_store.cuh", "r") as f:
    content = f.read()

content = content.replace("using SrcShape = decltype(src)::Layout::Shape;", "using SrcShape = typename decltype(src)::Layout::Shape;")
content = content.replace("using DstShape = decltype(dst2)::Layout::Shape;", "using DstShape = typename decltype(dst2)::Layout::Shape;")

with open("stage12_include/load_store.cuh", "w") as f:
    f.write(content)
