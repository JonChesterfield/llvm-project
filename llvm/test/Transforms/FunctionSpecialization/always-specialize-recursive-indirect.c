#include <stdbool.h>

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


