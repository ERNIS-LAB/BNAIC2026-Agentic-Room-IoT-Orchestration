#include "main.h"
#include "audio_record.h"
#include "b_u585i_iot02a_audio.h"
#include <math.h>
#include <stdint.h>
#include <limits.h>
#include <stdio.h>

#define AUDIO_INSTANCE        (0U)
#define SNAPSHOT_SAMPLES      (1024U)
#define BUFFER_SAMPLES        (SNAPSHOT_SAMPLES * 2U)
#define RECORD_BYTES          (BUFFER_SAMPLES * sizeof(int16_t))
#define RECORD_TIMEOUT_MS     (500U)
#define REF_AMPLITUDE         (32768.0f)
#define NUM_SNAPS             (10U)

static int16_t RecordBuff[BUFFER_SAMPLES] __attribute__((aligned(4)));

static volatile uint8_t HalfReady = 0;
static volatile uint8_t AudioError = 0;
static uint8_t audio_initialized = 0;

static float CalculateRMS(const int16_t *data, uint32_t length);
static float RMSTodBFS(float rms);
static void SortFloat(float *a, uint32_t n);
static int32_t CaptureOne_dBFS_x100(void);

// Main function to get dBFS
int32_t AudioRecord(void)
{
    float values[NUM_SNAPS];
    uint32_t valid = 0;

    // Capture NUM_SNAPS dBFS, so that we can select the median later on
    // One capture might not be representing real noise level
    for (uint32_t i = 0; i < NUM_SNAPS; i++)
    {
        int32_t v = CaptureOne_dBFS_x100();

        if (v != INT32_MIN)
        {
            values[valid++] = (float)v / 100.0f;
        }

        // Needs delay between records
        HAL_Delay(20);
    }

    if (valid == 0U)
        return INT32_MIN;

    // Sort the dBFS, so that we can get median later on
    SortFloat(values, valid);

    // Chooses the median to return
    float median;
    if ((valid & 1U) != 0U)
        median = values[valid / 2U];
    else
        median = 0.5f * (values[(valid / 2U) - 1U] + values[valid / 2U]);

    return (int32_t)(median*100.0f);
}

static int32_t CaptureOne_dBFS_x100(void)
{
    BSP_AUDIO_Init_t AudioInit;
    float rms, dBFS;
    uint32_t start;

    // Parameters required for the sensor
    AudioInit.Device        = AUDIO_IN_DEVICE_DIGITAL_MIC2;
    AudioInit.SampleRate    = AUDIO_FREQUENCY_16K;
    AudioInit.BitsPerSample = AUDIO_RESOLUTION_16B;
    AudioInit.ChannelsNbr   = 1;
    AudioInit.Volume        = 0;

    HalfReady  = 0;
    AudioError = 0;

    // Only initialise sensor if not initialized before
    if (audio_initialized == 0) {
    	uint32_t start = HAL_GetTick();
		int32_t initResult = BSP_AUDIO_IN_Init(AUDIO_INSTANCE, &AudioInit);
    	uint32_t finish = HAL_GetTick();
    	printf("Time taken for audio initialization: %lu\n", (unsigned long)(finish - start));
		if (initResult != BSP_ERROR_NONE)
		{
			return INT32_MIN;
		}
		audio_initialized = 1;
	}

    // Create recording and check if record is valid
    int32_t recResult = BSP_AUDIO_IN_Record(AUDIO_INSTANCE, (uint8_t *)RecordBuff, RECORD_BYTES);
    if (recResult != BSP_ERROR_NONE)
    {
        return INT32_MIN;
    }

    // Record the audio for RECORD_TIMEOUT_MS
    start = HAL_GetTick();
    while ((HalfReady == 0U) && (AudioError == 0U))
    {
        if ((HAL_GetTick() - start) > RECORD_TIMEOUT_MS)
        {
            BSP_AUDIO_IN_Stop(AUDIO_INSTANCE);
            return INT32_MIN;
        }
    }

    // Stop when unused
    BSP_AUDIO_IN_Stop(AUDIO_INSTANCE);

    if (AudioError != 0U)
    {
        return INT32_MIN;
    }

    // Convert recording to RMS to get average
    // then convert to dBFS before returned
    rms  = CalculateRMS(RecordBuff, SNAPSHOT_SAMPLES);
    dBFS = RMSTodBFS(rms);

    if (rms < 1.0f)
        return -99900;

    return (int32_t)(dBFS*100.0f);
}

// Conversion function from buffer to RMS
static float CalculateRMS(const int16_t *data, uint32_t length)
{
    float mean = 0.0f;
    float sum = 0.0f;

    for (uint32_t i = 0; i < length; i++)
        mean += (float)data[i];
    mean /= (float)length;

    for (uint32_t i = 0; i < length; i++)
    {
        float s = (float)data[i] - mean;
        sum += s * s;
    }

    return sqrtf(sum / (float)length);
}

// Conversion function from RMS to dBFS
static float RMSTodBFS(float rms)
{
    if (rms < 1.0f)
        return -999.0f;

    return 20.0f * log10f(rms / REF_AMPLITUDE);
}

// Bubble-sort implementation to retrieve median later on
static void SortFloat(float *a, uint32_t n)
{
    for (uint32_t i = 0; i < n; i++)
    {
        for (uint32_t j = i + 1; j < n; j++)
        {
            if (a[j] < a[i])
            {
                float t = a[i];
                a[i] = a[j];
                a[j] = t;
            }
        }
    }
}

void BSP_AUDIO_IN_HalfTransfer_CallBack(uint32_t Instance)
{
    (void)Instance;
    HalfReady = 1;
}

void BSP_AUDIO_IN_TransferComplete_CallBack(uint32_t Instance)
{
    (void)Instance;
}

void BSP_AUDIO_IN_Error_CallBack(uint32_t Instance)
{
    (void)Instance;
    AudioError = 1;
}
