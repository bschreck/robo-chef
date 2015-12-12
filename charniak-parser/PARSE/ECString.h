
#ifndef MECSTRING_H
#define MECSTRING_H

#define ECS gnu

#if ECS == gnu
using namespace std;
#include <string>
#define ECString string
#else
#include <bstring.h>
#define ECString string
#endif

#endif	/* ! MECSTRING_H */
