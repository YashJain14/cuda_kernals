#include <iostream>
#include <cute/tensor.hpp>

using namespace cute;

int main() {
    using SmemLayoutAtom = decltype(composition(Swizzle<3,3,3>{}, Layout<Shape<_8, _64>, Stride<_64, _1>>{}));
    using SmemLayoutKV = decltype(tile_to_shape(SmemLayoutAtom{}, Shape<Int<64>, Int<64>>{}));

    SmemLayoutKV layout_kv;
    auto layout_trans = make_layout(get<1>(layout_kv), get<0>(layout_kv));

    std::cout << "Original: " << layout_kv << std::endl;
    std::cout << "Transposed: " << layout_trans << std::endl;

    return 0;
}
