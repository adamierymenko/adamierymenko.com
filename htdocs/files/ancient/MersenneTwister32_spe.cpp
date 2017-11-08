//
// 32-bit Mersenne Twister PRNG for the IBM/Sony Cell Broadband Engine
// (c)2007 Adam Ierymenko [adam.ierymenko@gmail.com]
//
// This code is hereby released under the same terms as the original MT
// code on which it is based (see preserved notice below).
//
// Define USE_SPU_VECTOR to use the vector code. Otherwise a "naive"
// version is compiled using regular scalar code on the SPU.
//

// Comment out for the regular C implementation
#define USE_SPU_VECTOR

//
// Note: when USE_SPU_VECTOR is defined, the random number sequence will
// appear different. Upon closer inspection, you'll see that every fourth
// value is a value from the random stream from the regular non-vectorized
// version.  This is because the vectorized version essentially runs four
// seperate MT RNGs in parallel.  The seeding code seeds all four at once
// using different seeds based on the first seed.
//
// This shouldn't matter in practice, since you don't usually care about
// the characteristics of random numbers other than that they are random.
// But be aware that the sequence will not be reproducible between vector
// and non-vector versions.
//

/* 
   A C-program for MT19937, with initialization improved 2002/1/26.
   Coded by Takuji Nishimura and Makoto Matsumoto.

   Before using, initialize the state by using init_genrand(seed)  
   or init_by_array(init_key, key_length).

   Copyright (C) 1997 - 2002, Makoto Matsumoto and Takuji Nishimura,
   All rights reserved.                          

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions
   are met:

     1. Redistributions of source code must retain the above copyright
        notice, this list of conditions and the following disclaimer.

     2. Redistributions in binary form must reproduce the above copyright
        notice, this list of conditions and the following disclaimer in the
        documentation and/or other materials provided with the distribution.

     3. The names of its contributors may not be used to endorse or promote 
        products derived from this software without specific prior written 
        permission.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
   CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


   Any feedback is very welcome.
   http://www.math.sci.hiroshima-u.ac.jp/~m-mat/MT/emt.html
   email: m-mat @@ math.sci.hiroshima-u.ac.jp (remove space)
*/

#include <stdint.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <spu_intrinsics.h>

// Mersenne Twister parameters
#define _MT32_N 624
#define _MT32_M 397
#define _MT32_MATRIX_A 0x9908b0dfUL   /* constant vector a */
#define _MT32_UPPER_MASK 0x80000000UL /* most significant w-r bits */
#define _MT32_LOWER_MASK 0x7fffffffUL /* least significant r bits */

// This can be placed into an include file for use in a larger app...
class MersenneTwister32
{
  public:
    MersenneTwister32(uint32_t s);
    uint32_t operator()();

  private:
#ifdef USE_SPU_VECTOR
    vec_uint4 mt[_MT32_N]; // four concurrent twisters!
    vec_uint4 current;
    uintptr_t currentElement;
    uintptr_t mti;
#else
    uint32_t mt[_MT32_N];
    uintptr_t mti;
#endif
};

MersenneTwister32::MersenneTwister32(uint32_t s)
{
#ifdef USE_SPU_VECTOR
  // This could probably be made more efficient with some serious thought, but
  // init only happens once so I didn't worry too much about it. The multiply
  // is the kicker here... -Adam
  uint32_t tmp[_MT32_N];
  for(uintptr_t e=0;e<4;++e) {
    tmp[0] = s;
    for(uintptr_t i=1;i<_MT32_N;++i)
      tmp[i] = (1812433253UL * (tmp[i-1] ^ (tmp[i-1] >> 30)) + i);
    for(uintptr_t i=0;i<_MT32_N;++i)
      mt[i] = spu_insert(tmp[i],mt[i],e);
    s ^= tmp[_MT32_N - 1];
  }
  mti = _MT32_N;
  currentElement = 4;
#else
  mt[0] = s;
  for(mti=1;mti<_MT32_N;++mti)
    mt[mti] = (1812433253UL * (mt[mti-1] ^ (mt[mti-1] >> 30)) + mti);
#endif
}

uint32_t MersenneTwister32::operator()()
{
#ifdef USE_SPU_VECTOR
  if (mti >= _MT32_N) {
    register vec_uint4 x,y,z;
    register uintptr_t kk;
    for(kk=0;kk<_MT32_N-_MT32_M;++kk) {
      y = spu_and(mt[kk],(vec_uint4){ _MT32_UPPER_MASK,_MT32_UPPER_MASK,_MT32_UPPER_MASK,_MT32_UPPER_MASK });
      z = spu_and(mt[kk+1],(vec_uint4){ _MT32_LOWER_MASK,_MT32_LOWER_MASK,_MT32_LOWER_MASK,_MT32_LOWER_MASK });
      x = spu_or(y,z);
      y = spu_rlmask(x,-1);
      z = spu_maskw(spu_extract(spu_gather(x),0));
      z = spu_and(z,(vec_uint4){ _MT32_MATRIX_A,_MT32_MATRIX_A,_MT32_MATRIX_A,_MT32_MATRIX_A });
      x = spu_xor(mt[kk + _MT32_M],y);
      mt[kk] = spu_xor(x,z);
    }
    for(;kk<_MT32_N-1;++kk) {
      y = spu_and(mt[kk],(vec_uint4){ _MT32_UPPER_MASK,_MT32_UPPER_MASK,_MT32_UPPER_MASK,_MT32_UPPER_MASK });
      z = spu_and(mt[kk+1],(vec_uint4){ _MT32_LOWER_MASK,_MT32_LOWER_MASK,_MT32_LOWER_MASK,_MT32_LOWER_MASK });
      x = spu_or(y,z);
      y = spu_rlmask(x,-1);
      z = spu_maskw(spu_extract(spu_gather(x),0));
      z = spu_and(z,(vec_uint4){ _MT32_MATRIX_A,_MT32_MATRIX_A,_MT32_MATRIX_A,_MT32_MATRIX_A });
      x = spu_xor(mt[kk + (_MT32_M - _MT32_N)],y);
      mt[kk] = spu_xor(x,z);
    }
    y = spu_and(mt[_MT32_N - 1],(vec_uint4){ _MT32_UPPER_MASK,_MT32_UPPER_MASK,_MT32_UPPER_MASK,_MT32_UPPER_MASK });
    z = spu_and(mt[0],(vec_uint4){ _MT32_LOWER_MASK,_MT32_LOWER_MASK,_MT32_LOWER_MASK,_MT32_LOWER_MASK });
    x = spu_or(y,z);
    y = spu_rlmask(x,-1);
    z = spu_maskw(spu_extract(spu_gather(x),0));
    z = spu_and(z,(vec_uint4){ _MT32_MATRIX_A,_MT32_MATRIX_A,_MT32_MATRIX_A,_MT32_MATRIX_A });
    x = spu_xor(mt[_MT32_M - 1],y);
    mt[_MT32_N - 1] = spu_xor(x,z);
    mti = 0;
  }
#else
  if (mti >= _MT32_N) {
    uintptr_t kk;
    for(kk=0;kk<_MT32_N-_MT32_M;++kk) {
      uint32_t x = (mt[kk] & _MT32_UPPER_MASK) | (mt[kk+1] & _MT32_LOWER_MASK);
      mt[kk] = mt[kk + _MT32_M] ^ (x >> 1) ^ ((x & 1) * _MT32_MATRIX_A);
    }
    for(;kk<_MT32_N-1;++kk) {
      uint32_t x = (mt[kk] & _MT32_UPPER_MASK) | (mt[kk+1] & _MT32_LOWER_MASK);
      mt[kk] = mt[kk + (_MT32_M - _MT32_N)] ^ (x >> 1) ^ ((x & 1) * _MT32_MATRIX_A);
    }
    uint32_t x = (mt[_MT32_N - 1] & _MT32_UPPER_MASK) | (mt[0] & _MT32_LOWER_MASK);
    mt[_MT32_N - 1] = mt[_MT32_M - 1] ^ (x >> 1) ^ ((x & 1) * _MT32_MATRIX_A);
    mti = 0;
  }
#endif

#ifdef USE_SPU_VECTOR
  if (currentElement >= 4) {
    currentElement = 0;
    current = mt[mti++];
    current = spu_xor(current,spu_rlmask(current,-11));
    current = spu_xor(current,spu_and(spu_sl(current,7),(vec_uint4){ 0x9d2c5680UL,0x9d2c5680UL,0x9d2c5680UL,0x9d2c5680UL }));
    current = spu_xor(current,spu_and(spu_sl(current,15),(vec_uint4){ 0xefc60000UL,0xefc60000UL,0xefc60000UL,0xefc60000UL }));
    current = spu_xor(current,spu_rlmask(current,-18));
  }
  uint32_t y = spu_extract(current,currentElement++);
#else
  uint32_t y = mt[mti++];
  y ^= (y >> 11);
  y ^= (y << 7) & 0x9d2c5680UL;
  y ^= (y << 15) & 0xefc60000UL;
  y ^= (y >> 18);
#endif

  return y;
}

// SPU main function for testing and benchmarking...
int main(unsigned long long spe, unsigned long long argp, unsigned long long envp)
{
  struct timeval start,end;
  MersenneTwister32 prng(12345);

  for(int i=0;i<16;++i)
    printf("%.8x\n",prng());

  printf("\nBenchmarking...\n");
  gettimeofday(&start,(struct timezone *)0);
  uint32_t foo = 0;
  for(int i=0;i<1000000000;++i)
    foo ^= prng();
  gettimeofday(&end,(struct timezone *)0);
  printf("Result (not important): %.8x\n",foo);
  double duration = (((double)end.tv_sec) + (((double)end.tv_usec) / 1000000.0)) - (((double)start.tv_sec) + (((double)start.tv_usec) / 1000000.0));
  printf("time: %f seconds\n",duration);

  return 0;
}
