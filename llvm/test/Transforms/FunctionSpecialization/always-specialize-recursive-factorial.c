unsigned factorial(__attribute__((always_specialize)) unsigned x)
{
  if (x < 2) return 1;
  return x * factorial(x - 1);
}

unsigned factorial_driver()
{
  return factorial(0) + factorial(1) + factorial(2) + factorial(3) + factorial(4);
}

