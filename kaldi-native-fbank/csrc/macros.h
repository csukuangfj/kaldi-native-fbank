// kaldi-native-fbank/csrc/macros.h
//
// Copyright      2026  Xiaomi Corporation

#ifndef KALDI_NATIVE_FBANK_CSRC_MACROS_H_
#define KALDI_NATIVE_FBANK_CSRC_MACROS_H_

#include <stdint.h>
#include <stdio.h>

#if __ANDROID_API__ >= 8
#include "android/log.h"
#define KNF_LOGE(...)                                                         \
  do {                                                                        \
    fprintf(stderr, "%s:%s:%d ", __FILE__, __func__,                          \
            static_cast<int32_t>(__LINE__));                                  \
    fprintf(stderr, __VA_ARGS__);                                             \
    fprintf(stderr, "\n");                                                    \
    __android_log_print(ANDROID_LOG_WARN, "kaldi-native-fbank", "%s:%s:%d",   \
                        __FILE__, __func__, static_cast<int32_t>(__LINE__));  \
    __android_log_print(ANDROID_LOG_WARN, "kaldi-native-fbank", __VA_ARGS__); \
  } while (0)
#elif defined(__OHOS__)
#include "hilog/log.h"
#undef LOG_DOMAIN
#undef LOG_TAG
#define LOG_DOMAIN 0x6666
#define LOG_TAG "kaldi-native-fbank"
#define KNF_LOGE(...) OH_LOG_INFO(LOG_APP, __VA_ARGS__)
#else
/**
 * printf-style logging of errors/warnings: prints ``file:func:line`` and the
 * message to stderr.  Usage: ``KNF_LOGE("bad value: %d", x);``
 */
#define KNF_LOGE(...)                                \
  do {                                               \
    fprintf(stderr, "%s:%s:%d ", __FILE__, __func__, \
            static_cast<int32_t>(__LINE__));         \
    fprintf(stderr, __VA_ARGS__);                    \
    fprintf(stderr, "\n");                           \
  } while (0)
#endif

#endif  // KALDI_NATIVE_FBANK_CSRC_MACROS_H_
