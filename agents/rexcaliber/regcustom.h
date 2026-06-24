// src/include/regex/regcustom.h


/* headers if any */
#include "postgres.h"

#include <ctype.h>
#include <limits.h>

/*
 * towlower() and friends should be in <wctype.h>, but some pre-C99 systems
 * declare them in <wchar.h>.
 */
#ifdef HAVE_WCHAR_H
#include <wchar.h>
#endif
#ifdef HAVE_WCTYPE_H
#include <wctype.h>
#endif

#include "mb/pg_wchar.h"


/* overrides for regguts.h definitions, if any */
#define FUNCPTR(name, args) (*name) args
#define MALLOC(n)		malloc(n)
#define FREE(p)			free(VS(p))
#define REALLOC(p,n)	realloc(VS(p),n)
#define assert(x)		Assert(x)

/* internal character type and related */
typedef pg_wchar chr;			/* the type itself */
typedef unsigned uchr;			/* unsigned type that will hold a chr */
typedef int celt;				/* type to hold chr, or NOCELT */

#define NOCELT	(-1)			/* celt value which is not valid chr */
#define CHR(c)	((unsigned char) (c))	/* turn char literal into chr literal */
#define DIGITVAL(c) ((c)-'0')	/* turn chr digit into its value */
#define CHRBITS 32				/* bits in a chr; must not use sizeof */
#define CHR_MIN 0x00000000		/* smallest and largest chr; the value */
#define CHR_MAX 0xfffffffe		/* CHR_MAX-CHR_MIN+1 should fit in uchr */

/* functions operating on chr */
#define iscalnum(x) pg_wc_isalnum(x)
#define iscalpha(x) pg_wc_isalpha(x)
#define iscdigit(x) pg_wc_isdigit(x)
#define iscspace(x) pg_wc_isspace(x)

/* and pick up the standard header */
#include "regex.h"
