#include <stdbool.h>

// Needs to be O0 to not immediately turn into the undef
unsigned loop(__attribute__((always_specialize)) unsigned x)
{
  return loop(5) + loop(x);
}

unsigned loop_driver()
{
  return loop(5);
}
