

// Looking to have both call stacks end in the same specialised leaf.42.81 call
int leaf(__attribute__((always_specialize)) int x,
         __attribute__((always_specialize)) int y)
{
  return x + y;
}


int in_order(__attribute__((always_specialize)) int x)
{
  // create leaf.42.0()
  return leaf(42, x);
}

int swapped(__attribute__((always_specialize)) int x)
{
  // create leaf.0.81()
  return leaf(x, 81);
}

int root()
{
  // creates two specialisations which leaves in_order and swapped somewhat dead
  return in_order(81) * swapped(42);
}


