// RUN: %clang_cc1 -verify -fsyntax-only %s

extern "C"
int target(int x,
        __attribute__((always_specialize))  int y,
         __attribute__((always_specialize))  int z ) // expected-error {{'always_specialize' wombat}}
{
  if (z > 3)
    {
      return x * y + z;
    }
  else
    {
      return x / y;
    }
}


extern "C" int call_A(int x, int y)
{
  return target(x, 42, y);
}

extern "C" int call_B(int x)
{
  return target(x, 101, 98);
}

extern "C" int call_C(int x)
{
  return target(x, 42, 98);
}


extern "C" int call_Btwice(int x)
{
  return target(x, 101, 98) + target(x, 101, 98);
}

struct pair
{
  float x;
  float y;
};

extern pair pair_instance;

extern "C"
float used_pair(__attribute__((always_specialize)) pair *p)
{
  return p->x + p->y;
}

extern "C"
float call_used_pair()
{
  return used_pair(&pair_instance);
}


const struct nested
{
  int (*fptr)(int, int, int);
} nested_instance = {target};

extern "C"
int used_nested(__attribute__((always_specialize)) const nested *p, int x, int y)
{
  return p->fptr(x, y, 101);
}

extern "C"
int call_used_nested(int x, int y)
{
  return used_nested(&nested_instance, x, y);
}
