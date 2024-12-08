#include <stdarg.h>

// not libc names to avoid hitting any special magic there
int vprint(void*F, const char * fmt, va_list arg);

// want it to specialise on the format string and do something defensible on the ... pack
int print(void*F,
          __attribute__((always_specialize)) const char * fmt,
          __attribute__((always_specialize)) ...)
{
  int ret;
  va_list va;
  va_start(va, fmt);
  ret = vprint(F, fmt, va);
  va_end(va);
  return ret;
}


