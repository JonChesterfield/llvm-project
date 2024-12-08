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

unsigned factorial(__attribute__((always_specialize)) unsigned x)
{
  if (x < 2) return 1;
  return x * factorial(x - 1);
}


unsigned factorial_driver()
{
  return factorial(0) + factorial(1) + factorial(2) + factorial(3) + factorial(4);
}



bool even(__attribute__((always_specialize)) unsigned x);

bool odd(__attribute__((always_specialize)) unsigned x)
{
  return x == 0 ? false : even(x-1);
}

bool even(unsigned x)
{
  return x == 0 ? true : odd(x-1);
}

bool evenodd_driver()
{
  return even(0) && !even(1) && even(2) && !even(3) && !odd(0) && odd(1) && !odd(2) && odd(3);
}

