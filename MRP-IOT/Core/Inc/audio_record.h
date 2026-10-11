#ifndef AUDIO_RECORD_H
#define AUDIO_RECORD_H

#ifdef __cplusplus
extern "C" {
#endif

#include "main.h"
#include <stdint.h>

/* Public function ----------------------------------------------------------*/
int32_t AudioRecord(void);

/* BSP audio callbacks ------------------------------------------------------*/
void BSP_AUDIO_IN_HalfTransfer_CallBack(uint32_t Instance);
void BSP_AUDIO_IN_TransferComplete_CallBack(uint32_t Instance);
void BSP_AUDIO_IN_Error_CallBack(uint32_t Instance);

#ifdef __cplusplus
}
#endif

#endif /* AUDIO_RECORD_H */
