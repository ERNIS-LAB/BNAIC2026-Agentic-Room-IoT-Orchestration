/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file    app_netxduo.h
  * @author  MCD Application Team
  * @brief   NetXDuo applicative header file
  ******************************************************************************
    * @attention
  *
  * Copyright (c) 2026 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */
/* USER CODE END Header */
/* Define to prevent recursive inclusion -------------------------------------*/
#ifndef __APP_NETXDUO_H__
#define __APP_NETXDUO_H__

#ifdef __cplusplus
extern "C" {
#endif

/* Includes ------------------------------------------------------------------*/
#include "nx_api.h"

/* Private includes ----------------------------------------------------------*/

/* USER CODE BEGIN Includes */
#include "main.h"
#include "network_config.h"
#include "nxd_dhcp_client.h"
#include "nxd_mqtt_client.h"
#include "nx_driver_emw3080.h"
/* USER CODE END Includes */

/* Exported types ------------------------------------------------------------*/
/* USER CODE BEGIN ET */

/* USER CODE END ET */

/* Exported constants --------------------------------------------------------*/
/* USER CODE BEGIN EC */

  /* Threads configuration */
#define PAYLOAD_SIZE                1544
#ifdef CAMERA_HD_ABOVE
	#define NX_PACKET_POOL_SIZE         (( PAYLOAD_SIZE + sizeof(NX_PACKET)) * 270)
#else
	#define NX_PACKET_POOL_SIZE         (( PAYLOAD_SIZE + sizeof(NX_PACKET)) * 150)
#endif
#define DEFAULT_MEMORY_SIZE         1024
#define ARP_MEMORY_SIZE             DEFAULT_MEMORY_SIZE
#define DEFAULT_MAIN_PRIORITY       10
#define MQTT_PRIORITY               12
#define THREAD_MEMORY_SIZE          4 * DEFAULT_MEMORY_SIZE
#define APP_QUEUE_SIZE              10

/* MQTT Client configuration */
#define MQTT_CLIENT_STACK_SIZE      1024 * 10
#define MQTT_THREAD_PRIORTY         2
#define CLEAN_SESSION               NX_TRUE
#define STRLEN(p)                   (sizeof(p) - 1)

/* NODE_NAME — change this for each board before building:
 *   Board 1:  #ifndef NODE_NAME
#define NODE_NAME  "r2001/n1"
#endif
 *   Board 2:  #ifndef NODE_NAME
#define NODE_NAME  "r1001/n1"
#endif
 *   Keep it short so all request topics fit in the 32-byte firmware buffer. */
#ifndef NODE_NAME
#define NODE_NAME					"rls/n1"
#endif

#define NODE_ID_REQ_TOPIC_NAME		"node/req"
#define NODE_ID_RES_TOPIC_NAME		"node/res"

#define SUB_TOPIC_NAME			    NODE_NAME "/req/+"
#define CAMERA_REQ_TOPIC_NAME		NODE_NAME "/req/camera"
#define MIC_REQ_TOPIC_NAME			NODE_NAME "/req/mic"
#define TEMPERATURE_REQ_TOPIC_NAME	NODE_NAME "/req/temperature"
#define LIGHT_REQ_TOPIC_NAME		NODE_NAME "/req/light"
#define CAMERA_RES_TOPIC_NAME		NODE_NAME "/res/camera"
#define MIC_RES_TOPIC_NAME			NODE_NAME "/res/mic"
#define TEMPERATURE_RES_TOPIC_NAME	NODE_NAME "/res/temperature"
#define LIGHT_RES_TOPIC_NAME		NODE_NAME "/res/light"


#define MQTT_PORT                   1883

#define QOS0                        0
#define QOS1                        1

#define DEMO_MESSAGE_EVENT          1
#define DEMO_ALL_EVENTS             3

#define NULL_ADDRESS                0
#define DEFAULT_TIMEOUT             10 * NX_IP_PERIODIC_RATE
/* USER CODE END EC */

/* Exported macro ------------------------------------------------------------*/
/* USER CODE BEGIN EM */
#define PRINT_IP_ADDRESS(addr)           do { \
                                              printf("STM32 %s: %lu.%lu.%lu.%lu \n", #addr, \
                                                (addr >> 24) & 0xff,                        \
                                                  (addr >> 16) & 0xff,                      \
                                                    (addr >> 8) & 0xff,                     \
                                                      (addr & 0xff));                       \
                                            } while(0)

/* USER CODE END EM */

/* Exported functions prototypes ---------------------------------------------*/
UINT MX_NetXDuo_Init(VOID *memory_ptr);

/* USER CODE BEGIN EFP */

/* USER CODE END EFP */

/* Private defines -----------------------------------------------------------*/
/* USER CODE BEGIN PD */

/* USER CODE END PD */

/* USER CODE BEGIN 1 */

/* USER CODE END 1 */

#ifdef __cplusplus
}
#endif
#endif /* __APP_NETXDUO_H__ */
