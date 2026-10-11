#include "main.h"
#include "b_u585i_iot02a_env_sensors.h"
#include "env_sensor.h"
#include <stdint.h>
#include <stdio.h>

static uint8_t env_initialized = 0;

int32_t TempRecord(void)
{
    float temperature;
    float sum = 0;
    int count = 0;

    // Initialize sensor if not initialized before
    if (env_initialized == 0) {
    	uint32_t start = HAL_GetTick();
    	int32_t ret = BSP_ENV_SENSOR_Init(0, ENV_TEMPERATURE);
    	uint32_t finish = HAL_GetTick();
    	printf("Time taken for temperature initialization: %lu\n", (unsigned long)(finish - start));
		if (ret != BSP_ERROR_NONE) {
			return INT32_MIN;
		}
		env_initialized = 1;
	}

    // Enable the sensor and set a delay to make sure
    // sensor is ready to be used
    BSP_ENV_SENSOR_Enable(0, ENV_TEMPERATURE);
    BSP_ENV_SENSOR_SetOutputDataRate(0, ENV_TEMPERATURE, 1.0f);
    HAL_Delay(200);

    // Collect 10 temperature readings with a delay between them.
    // Required for accurate readings
    for (int i = 0; i < 10; i++) {
        if (BSP_ENV_SENSOR_GetValue(0, ENV_TEMPERATURE, &temperature) == BSP_ERROR_NONE) {
            sum += temperature;
            count++;
        }
        HAL_Delay(500);
    }

    // Disable the sensor when not used
    BSP_ENV_SENSOR_Disable(0, ENV_TEMPERATURE);

    // Return the average measured temperature if any. Otherwise INT32_MIN
    if (count == 0) return INT32_MIN;
    return (int32_t)(sum / count);
}

