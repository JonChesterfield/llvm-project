int callee(__attribute__((always_specialize)) int x,
            int y,
            __attribute__((always_specialize)) int z)
{
  return (x + z) * z;
}

int first(int a, int b)
{
  return callee(42, a, b);
}

int second(int a, int b)
{
  return callee(a, 42, b);
}

int third(int a, int b)
{
  return callee(a, b, 42);
}

int both(int a)
{
  return callee(21, a, 42);
}


int ptrcallee(__attribute__((always_specialize)) int *x,
              int *y,
              __attribute__((always_specialize)) int *z)
{
  return (*x + *z) * (*z);
}

int ptrfirst(int *a, int *b)
{
  static int x = 42;
  return ptrcallee(&x, a, b);
}

int ptrboth(int *a)
{
  static int x = 42;
  static const int y = 81;
  return ptrcallee(&x, a, &y);
}

int ptrallsame(void)
{
  static int x = 42;
  return ptrcallee(&x, &x, &x);
}

int virtualcallee(int);

int virtualcall( __attribute__((always_specialize)) int (*func)(int), int x)
{
  return func(x);
}

int devirtualisecaller( int x)
{
  return virtualcall(virtualcallee, x);
}


