#include "test/UnitTest/Test.h"

TEST(LlvmLibcWot, second)
{
  ASSERT_EQ(0u, __builtin_amdgcn_grid_size_x());
  ASSERT_EQ(4, 3);
}
