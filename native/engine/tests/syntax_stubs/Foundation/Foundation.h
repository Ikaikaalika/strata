// Syntax-check stub for Linux CI only: declares the Foundation surface the
// engine uses so clang can parse Objective-C++ without Apple SDKs.
#pragma once
#include <stddef.h>
typedef unsigned long NSUInteger;
typedef long NSInteger;
typedef signed char BOOL;
#define nil nullptr
@interface NSObject
+ (instancetype)new;
@end
@interface NSString : NSObject
+ (instancetype)stringWithUTF8String:(const char*)s;
- (const char*)UTF8String;
@end
@interface NSError : NSObject
- (NSString*)localizedDescription;
@end
