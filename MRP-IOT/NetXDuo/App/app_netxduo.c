/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file    app_netxduo.c
  * @author  MCD Application Team
  * @brief   NetXDuo applicative file
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

/* Includes ------------------------------------------------------------------*/
#include "app_netxduo.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include "app_azure_rtos.h"
#include "nx_ip.h"
#include "pb_encode.h"
#include "pb_decode.h"
#include "pb_common.h"
#include "iot.pb.h"
#include "env_sensor.h"
#include "b_u585i_iot02a_camera.h"
#include "audio_record.h"
#include "b_u585i_iot02a_light_sensor.h"
/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */
TX_THREAD AppMainThread;
TX_THREAD AppMQTTClientThread;
TX_QUEUE  MsgQueueOne;
TX_SEMAPHORE Semaphore;
NX_PACKET_POOL  AppPool;
NX_IP           IpInstance;
NX_DHCP         DhcpClient;
NXD_MQTT_CLIENT MqttClient;
ULONG   IpAddress;
ULONG   NetMask;
ULONG mqtt_client_stack[MQTT_CLIENT_STACK_SIZE];
TX_EVENT_FLAGS_GROUP mqtt_app_flag;
/* Declare buffers to hold message and topic. */
static UCHAR message_buffer[16];
static UCHAR topic_buffer[32];
static uint8_t cam_initialized = 0;
static uint8_t light_initialized = 0;
static uint8_t disconnected = 0;
/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */
#define IPV4_OCTETS 4
#define MAX_CAMERA_BYTES 256000
#define MQTT_KEEPALIVE          60
/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */
/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/
/* USER CODE BEGIN PV */
volatile uint8_t frameFlag;
static uint8_t cameraBuffer[MAX_CAMERA_BYTES];
static uint32_t pResult[2];

#ifndef CAMERA_HD_ABOVE
static uint8_t protoBuffer[160000];
#endif
/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
/* USER CODE BEGIN PFP */
static VOID App_Main_Thread_Entry(ULONG thread_input);
static VOID App_MQTT_Client_Thread_Entry(ULONG thread_input);
static VOID ip_address_change_notify_callback(NX_IP   *  ip_instance, VOID   *  ptr);
/* USER CODE END PFP */

/**
  * @brief  Application NetXDuo Initialization.
  * @param memory_ptr: memory pointer
  * @retval int
  */
UINT MX_NetXDuo_Init(VOID *memory_ptr)
{
  UINT ret = NX_SUCCESS;
  TX_BYTE_POOL *byte_pool = (TX_BYTE_POOL*)memory_ptr;
   /* USER CODE BEGIN App_NetXDuo_MEM_POOL */
    (void)byte_pool;
  /* USER CODE END App_NetXDuo_MEM_POOL */
  /* USER CODE BEGIN 0 */
  /* USER CODE END 0 */

  /* USER CODE BEGIN MX_NetXDuo_Init */
#if (USE_STATIC_ALLOCATION == 1)
    CHAR   *  pointer;
    /* Initialize the NetX system. */
    nx_system_initialize();
    /* Allocate the memory for packet_pool.  */
    if (tx_byte_allocate(byte_pool, (VOID **) &pointer, NX_PACKET_POOL_SIZE, TX_NO_WAIT) != TX_SUCCESS) {
        printf("Error detected\n");
        return TX_POOL_ERROR;
    }

    /* Create the Packet pool to be used for packet allocation */
    ret = nx_packet_pool_create(&AppPool, "Main Packet Pool", PAYLOAD_SIZE, pointer, NX_PACKET_POOL_SIZE);
    if (ret != NX_SUCCESS) {
        printf("Error detected\n");
        return NX_NOT_ENABLED;
    }

    /* Allocate the memory for Ip_Instance */
    if (tx_byte_allocate(byte_pool, (VOID **) &pointer, 2*DEFAULT_MEMORY_SIZE, TX_NO_WAIT) != TX_SUCCESS) {
        printf("Error detected\n");
        return TX_POOL_ERROR;
    }

    /* Create the main NX_IP instance */
    ret = nx_ip_create(&IpInstance, "Main Ip instance", NULL_ADDRESS, NULL_ADDRESS, &AppPool, nx_driver_emw3080_entry, pointer, 2   *   DEFAULT_MEMORY_SIZE, DEFAULT_MAIN_PRIORITY);
    if (ret != NX_SUCCESS) {
        printf("Error detected\n");
        return NX_NOT_ENABLED;
    }

    /* create the DHCP client */
    ret = nx_dhcp_create(&DhcpClient, &IpInstance, "DHCP Client");
    if (ret != NX_SUCCESS) {
        printf("Error detected\n");
        return NX_NOT_ENABLED;
    }

    /* Allocate the memory for ARP */
    if (tx_byte_allocate(byte_pool, (VOID **) &pointer, ARP_MEMORY_SIZE, TX_NO_WAIT) != TX_SUCCESS) {
        printf("Error detected\n");
        return TX_POOL_ERROR;
    }

    /* Enable the ARP protocol and provide the ARP cache size for the IP instance */
    ret = nx_arp_enable(&IpInstance, (VOID   *  )pointer, ARP_MEMORY_SIZE);
    if (ret != NX_SUCCESS) {
        printf("Error detected\n");
        return NX_NOT_ENABLED;
    }

    /* Enable the UDP protocol required for DHCP communication */
    ret = nx_udp_enable(&IpInstance);
    if (ret != NX_SUCCESS) {
        printf("Error detected\n");
        return NX_NOT_ENABLED;
    }

    /* Enable the TCP protocol required for DNS, MQTT.. */
    ret = nx_tcp_enable(&IpInstance);
    if (ret != NX_SUCCESS) {
        printf("Error detected\n");
        return NX_NOT_ENABLED;
    }

    /* Allocate the memory for main thread   */
    if (tx_byte_allocate(byte_pool, (VOID **) &pointer, THREAD_MEMORY_SIZE, TX_NO_WAIT) != TX_SUCCESS) {
        printf("Error detected\n");
        return TX_POOL_ERROR;
    }

    /* Create the main thread */
    ret = tx_thread_create(&AppMainThread, "App Main thread", App_Main_Thread_Entry, 0, pointer, THREAD_MEMORY_SIZE, DEFAULT_MAIN_PRIORITY, DEFAULT_MAIN_PRIORITY, TX_NO_TIME_SLICE, TX_AUTO_START);
    if (ret != TX_SUCCESS) {
        printf("Error detected\n");
        return NX_NOT_ENABLED;
    }

    /* Allocate the memory for MQTT client thread   */
    if (tx_byte_allocate(byte_pool, (VOID **) &pointer, THREAD_MEMORY_SIZE, TX_NO_WAIT) != TX_SUCCESS) {
        printf("Error detected\n");
        return TX_POOL_ERROR;
    }

    /* create the MQTT client thread */
    ret = tx_thread_create(&AppMQTTClientThread, "App MQTT Thread", App_MQTT_Client_Thread_Entry, 0, pointer, THREAD_MEMORY_SIZE, MQTT_PRIORITY, MQTT_PRIORITY, TX_NO_TIME_SLICE, TX_DONT_START);
    if (ret != TX_SUCCESS) {
        printf("Error detected\n");
        return NX_NOT_ENABLED;
    }

    /* Allocate the MsgQueueOne.  */
    if (tx_byte_allocate(byte_pool, (VOID **) &pointer, APP_QUEUE_SIZE  *  sizeof(ULONG), TX_NO_WAIT) != TX_SUCCESS) {
        printf("Error detected\n");
        ret = TX_POOL_ERROR;
    }

    /* Create the MsgQueueOne shared by MsgSenderThreadOne and MsgReceiverThread */
    if (tx_queue_create(&MsgQueueOne, "Message Queue One", TX_1_ULONG, pointer, APP_QUEUE_SIZE  *  sizeof(ULONG)) != TX_SUCCESS) {
        printf("Error detected\n");
        ret = TX_QUEUE_ERROR;
    }

    /* set DHCP notification callback  */
    tx_semaphore_create(&Semaphore, "DHCP Semaphore", 0);

#endif
  /* USER CODE END MX_NetXDuo_Init */

  return ret;
}

/* USER CODE BEGIN 1 */
void BSP_CAMERA_FrameEventCallback(uint32_t Instance)
{
	frameFlag = 1;
}

/**
* @brief  ip address change callback.
* @param ip_instance: NX_IP instance
* @param ptr: user data
* @retval none
*/
static VOID ip_address_change_notify_callback(NX_IP   *  ip_instance, VOID   *  ptr) {
    /* release the semaphore as soon as an IP address is available */
    tx_semaphore_put(&Semaphore);
}

/**
* @brief  Main thread entry.
* @param thread_input: ULONG user argument used by the thread entry
* @retval none
*/
static VOID App_Main_Thread_Entry(ULONG thread_input) {
    UINT ret = NX_SUCCESS;
    ret = nx_ip_address_change_notify(&IpInstance, ip_address_change_notify_callback, NULL);
    if (ret != NX_SUCCESS) {
        printf("Error detected\n");
        Error_Handler();
    }

    /* start DHCP client */
    ret = nx_dhcp_start(&DhcpClient);
    if (ret != NX_SUCCESS) {
        printf("Error detected\n");
        Error_Handler();
    }

    /* wait until an IP address is ready */
    if (tx_semaphore_get(&Semaphore, TX_WAIT_FOREVER) != TX_SUCCESS) {
        printf("Error detected\n");
        Error_Handler();
    }

    ret = nx_ip_address_get(&IpInstance, &IpAddress, &NetMask);
    if (ret != TX_SUCCESS) {
        printf("Error detected\n");
        Error_Handler();
    }

    PRINT_IP_ADDRESS(IpAddress);
    /* start the MQTT client thread */
    tx_thread_resume(&AppMQTTClientThread);
    /* this thread is not needed any more, we relinquish it */
    tx_thread_relinquish();
    return;
}

/* Declare the disconnect notify function. */
static VOID my_disconnect_func(NXD_MQTT_CLIENT   *  client_ptr) {
    NX_PARAMETER_NOT_USED(client_ptr);

    // Print to UART if MQTT client disconnected. Also set flag, so
    // that we can reconnect to it later
    printf("client disconnected from broker < %s >.\n", MQTT_BROKER_NAME);
    disconnected = 1;

}

/* Declare the notify function. */
static VOID my_notify_func(NXD_MQTT_CLIENT  *   client_ptr, UINT number_of_messages) {
    NX_PARAMETER_NOT_USED(client_ptr);
    NX_PARAMETER_NOT_USED(number_of_messages);
    tx_event_flags_set(&mqtt_app_flag, DEMO_MESSAGE_EVENT, TX_OR);
    return;
}

// Converts an ipv4 string like "192.168.1.0" to ULONG which is
// required for the mqtt connect function
int parse_ipv4_string(char* ipv4_str, ULONG* ipv4) {
    char* octets_str = strtok(ipv4_str, ".");
    int octets[IPV4_OCTETS];

    for(int i = 0; i < IPV4_OCTETS; i++) {
    	if (octets_str == NULL) {
    		return 1;
    	}

        octets[i] = atoi(octets_str);
        octets_str = strtok(NULL, ".");
    }

    *ipv4 = IP_ADDRESS(octets[0], octets[1], octets[2], octets[3]);

    return 0;
}

// In-place JPEG image correction function
// Image sometimes have artifacts, this function
// removes these artifacts.
int get_corrected_jpeg_and_size(uint8_t* image) {
	if (image == NULL) {
		return 0;
	}

	if (image[0] != 0xFF || image[1] != 0xD8) {
		return 0;
	}

	int new_len = 2;
	for(int i = 2; i < MAX_CAMERA_BYTES - 1; i++) {

		// The camera sometimes have 16 consecutive zeros (0x00)
		// This leads to weird random patterns on the jpeg image
		// This just basically removes that
		if (i % 16 == 0 && i + 16 <= MAX_CAMERA_BYTES) {
			int num_consec_zeros = 0;
			for(int j = i; j < i + 16; j++) {
				if (image[j] == 0) {
					num_consec_zeros++;
				}
			}

			if (num_consec_zeros == 16) {
				i += 15;
				continue;
			}
		}

		// In-place, we don't want to waste memory so we rewrite
		// to same buffer immediately
		image[new_len] = image[i];
		new_len++;
		if (image[i] == 0xFF && image[i + 1] == 0xD9) {
			image[new_len] = image[i + 1];
			printf("Full size %d\n", i + 1);
			return new_len + 1;
		}
	}

	printf("Image was not found!\n");
	return 0;
}

int capture_image() {
	int image_size = 0;

	// Set buffer to all zero's, so that we can determine the artifacts
	// after we captured the image
	memset(cameraBuffer, 0, MAX_CAMERA_BYTES);
	frameFlag = 0;

	// Take the image
	int ret = BSP_CAMERA_Start(0, (uint8_t*)cameraBuffer, CAMERA_MODE_SNAPSHOT);
	if (ret != BSP_ERROR_NONE) {
		printf("Error occured while taking picture!\n");
		return image_size;
	}
	BSP_CAMERA_Stop(0);

	// Correct it if necessary
	image_size = get_corrected_jpeg_and_size(cameraBuffer);
	if (image_size == 0) {
		printf("Invalid JPEG captured!\n");
		HAL_Delay(100);
		return image_size;
	}

	return image_size;
}

// Temporary struct needed for protobuf callback
typedef struct {
	uint8_t* camBuffer;
	int size;
} cam_cb_arg_t;

// Using callback here, because otherwise protoc will generate code
// to reserve like an image worth of RAM on the stack which would mean
// we need a lot of RAM...
bool camera_write_callback(pb_ostream_t* stream, const pb_field_iter_t* field, void* const* arg) {
	cam_cb_arg_t* args = (cam_cb_arg_t*) *arg;

	if(!pb_encode_tag_for_field(stream, field)) {
		return false;
	}

	bool ret = pb_encode_string(stream, args->camBuffer, args->size);
	return ret;
}

// Default callback function for protobuffers for string types
bool str_write_callback(pb_ostream_t* stream, const pb_field_iter_t* field, void* const* arg) {
	const char* str = (const char*) *arg;
	if(!pb_encode_tag_for_field(stream, field)) {
		return false;
	}

	bool ret = pb_encode_string(stream, (uint8_t*) str, strlen(str));
	return ret;
}
/**
* @brief  MQTT Client thread entry.
* @param thread_input: ULONG user argument used by the thread entry
* @retval none
*/
static VOID App_MQTT_Client_Thread_Entry(ULONG thread_input) {
    UINT ret = NX_SUCCESS;
    NXD_ADDRESS mqtt_server_ip;
    ULONG events;
    UINT topic_length, message_length;

    ULONG mqtt_ipv4;
    if (parse_ipv4_string(MQTT_BROKER_NAME, &mqtt_ipv4) != 0) {
        printf("Error detected\n");
        Error_Handler();
    }

    mqtt_server_ip.nxd_ip_version = 4;
    mqtt_server_ip.nxd_ip_address.v4 = mqtt_ipv4;
    /* Create MQTT client instance. */
    ret = nxd_mqtt_client_create(&MqttClient, NODE_NAME, NODE_NAME, STRLEN(NODE_NAME), &IpInstance, &AppPool, (VOID*)mqtt_client_stack, MQTT_CLIENT_STACK_SIZE, MQTT_THREAD_PRIORTY, NX_NULL, 0);
    if (ret != NX_SUCCESS) {
        printf("Error detected\n");
        Error_Handler();
    }

    /* Register the disconnect notification function. */
    nxd_mqtt_client_disconnect_notify_set(&MqttClient, my_disconnect_func);
    /* Set the receive notify function. */
    nxd_mqtt_client_receive_notify_set(&MqttClient, my_notify_func);
    /* Create an MQTT flag */
    ret = tx_event_flags_create(&mqtt_app_flag, "my app event");
    if (ret != TX_SUCCESS) {
        printf("Error detected\n");
        Error_Handler();
    }

    // Leaving this code in so that we have at least MQTT auth if needed.
    //
    //  if (ret != NX_SUCCESS)
    // {
    //    Error_Handler();
    //  }

    /* Outer reconnection loop — keeps the node connected indefinitely. */
    while (1) {

        /* Connect to the MQTT broker. */
        ret = nxd_mqtt_client_connect(&MqttClient, &mqtt_server_ip, MQTT_PORT, MQTT_KEEPALIVE, CLEAN_SESSION, NX_WAIT_FOREVER);
        if (ret != NX_SUCCESS) {
            printf("\nMQTT client failed to connect to broker < %s >, retrying in 5 s...\n", MQTT_BROKER_NAME);
            tx_thread_sleep(5 * NX_IP_PERIODIC_RATE);
            continue;
        }
        printf("\nMQTT client connected to broker < %s > at PORT %d\n", MQTT_BROKER_NAME, MQTT_PORT);
        disconnected = 0;

        /* Subscribe to sensor request topics. */
        ret = nxd_mqtt_client_subscribe(&MqttClient, SUB_TOPIC_NAME, STRLEN(SUB_TOPIC_NAME), QOS1);
        if (ret != NX_SUCCESS) {
            printf("Subscribe to %s failed, reconnecting...\n", SUB_TOPIC_NAME);
            nxd_mqtt_client_disconnect(&MqttClient);
            tx_thread_sleep(2 * NX_IP_PERIODIC_RATE);
            continue;
        }

        /* Subscribe to node discovery topic. */
        ret = nxd_mqtt_client_subscribe(&MqttClient, NODE_ID_REQ_TOPIC_NAME, STRLEN(NODE_ID_REQ_TOPIC_NAME), QOS1);
        if (ret != NX_SUCCESS) {
            printf("Subscribe to %s failed, reconnecting...\n", NODE_ID_REQ_TOPIC_NAME);
            nxd_mqtt_client_disconnect(&MqttClient);
            tx_thread_sleep(2 * NX_IP_PERIODIC_RATE);
            continue;
        }

        /* Inner message-processing loop. */
        for (;;) {
            ret = nxd_mqtt_client_message_get(&MqttClient, topic_buffer, sizeof(topic_buffer), &topic_length, message_buffer, sizeof(message_buffer), &message_length);
            if (ret != NXD_MQTT_SUCCESS) {
                /* Break out to reconnect if the connection was lost. */
            	if(disconnected == 1) {
            		break;
            	}
                continue;
            }

        uint32_t millis = HAL_GetTick();
        printf("[%lu] Message received: TOPIC = \"%s\", MESSAGE = \"%s\"\n", (unsigned long)millis, topic_buffer, message_buffer);
        if (strcmp((char*) topic_buffer, TEMPERATURE_REQ_TOPIC_NAME) == 0) {

        	// We have to reset buffer, because if we have "temperature" first
        	// and then "camera", then it would become "cameraature"
        	memset(topic_buffer, 0, sizeof(topic_buffer));

        	// Buffer for encoded message, init the message to zero
        	// and create a protobuffer stream
            uint8_t temperatureBuffer[iot_Temperature_size];
            size_t message_length; {
                iot_Temperature message = iot_Temperature_init_zero;
                pb_ostream_t stream = pb_ostream_from_buffer(temperatureBuffer, sizeof(temperatureBuffer));
                uint32_t start = HAL_GetTick();
                // Get the temperature from the sensor
                int32_t temperature = TempRecord();
                uint32_t finish = HAL_GetTick();
                printf("Time taken for temperature recording: %lu\n", (unsigned long)(finish - start));
                message.temperature = temperature;

                start = HAL_GetTick();
                // Actual message encoding step
                bool status = pb_encode(&stream, iot_Temperature_fields, &message);
                finish = HAL_GetTick();
                printf("Time taken for temperature protobuf encoding: %lu\n", (unsigned long)(finish - start));

                // Store the message_length as it's needed for the publish so that MQTT
                // broker and other clients can know length of message + so we don't overtransmit
                // our buffer and leak info or get segmentation issues
                message_length = stream.bytes_written;
                if (!status) {
                    printf("Error while encoding message.\n");
                    continue;
                }

                // Actual publishing
                ret = nxd_mqtt_client_publish(&MqttClient, TEMPERATURE_RES_TOPIC_NAME, STRLEN(TEMPERATURE_RES_TOPIC_NAME), (CHAR*)temperatureBuffer, message_length, NX_TRUE, QOS1, NX_WAIT_FOREVER);
                if (ret != NX_SUCCESS) {
                    printf("Failed to publish\n");
                    continue;
                }

                tx_event_flags_get(&mqtt_app_flag, DEMO_ALL_EVENTS, TX_OR_CLEAR, &events, TX_WAIT_FOREVER);
            }

        } else if (strcmp((char*) topic_buffer, CAMERA_REQ_TOPIC_NAME) == 0) {
        	// Code is pretty much the same as first topic, so
        	// leaving the repetitive stuff uncommented.
        	memset(topic_buffer, 0, sizeof(topic_buffer));
        	int32_t ret;

        	// Initialize camera if needed with 1000ms delay to make sure
        	// camera is fully initialized. Experimented with 720p with custom drivers
        	// but not stable enough so behind flag
        	if (cam_initialized == 0) {
        		uint32_t start = HAL_GetTick();
#ifdef CAMERA_HD_ABOVE
        		ret = BSP_CAMERA_Init(0, CAMERA_R1280x720, OV5640_JPEG);
#else
        		ret = BSP_CAMERA_Init(0, CAMERA_R800x480, OV5640_JPEG);
#endif
        		uint32_t finish = HAL_GetTick();
        		printf("Time taken for initialization: %lu\n", (unsigned long)(finish - start));
				if (ret != BSP_ERROR_NONE) {
					printf("Error occured while initializing camera!\n");
					continue;
				}
				HAL_Delay(1000);
				cam_initialized = 1;
        	}

        	uint32_t start = HAL_GetTick();
        	int image_size = capture_image();
        	uint32_t finish = HAL_GetTick();
        	printf("Time taken for image capturing + processing: %lu\n", (unsigned long)(finish - start));
        	if (image_size == 0) {
        		printf("Unable to capture image\n");
        		continue;
        	}

			printf("Taken picture succesfully with size %d!\n", image_size);
			// If we use 720p, we cannot use protocol buffers, because not enough RAM.
#ifdef CAMERA_HD_ABOVE
			ret = nxd_mqtt_client_publish(&MqttClient, CAMERA_RES_TOPIC_NAME, STRLEN(CAMERA_RES_TOPIC_NAME), (CHAR*)cameraBuffer, image_size, NX_TRUE, QOS1, NX_WAIT_FOREVER);
			if (ret != NX_SUCCESS) {
				printf("Failed to publish\n");
				continue;
			}

			tx_event_flags_get(&mqtt_app_flag, DEMO_ALL_EVENTS, TX_OR_CLEAR, &events, TX_WAIT_FOREVER);
#else
         	size_t message_length; {
        	    iot_Camera message = iot_Camera_init_zero;
        	    cam_cb_arg_t args = {
        	    		.camBuffer = cameraBuffer,
						.size = image_size
        	    };
        	    message.camera.funcs.encode = camera_write_callback;
        	    message.camera.arg = &args;

        	    pb_ostream_t stream = pb_ostream_from_buffer(protoBuffer, sizeof(protoBuffer));

        	    start = HAL_GetTick();
        	    bool status = pb_encode(&stream, iot_Camera_fields, &message);
        	    finish = HAL_GetTick();
        	    printf("Time taken for camera protobuf encoding: %lu\n", (unsigned long)(finish - start));

        	    message_length = stream.bytes_written;
        	    if (!status) {
        	        printf("Error while encoding message.\n");
        	        continue;
        	    }
        	    printf("Publishing now!\n");
        	    ret = nxd_mqtt_client_publish(&MqttClient, CAMERA_RES_TOPIC_NAME, STRLEN(CAMERA_RES_TOPIC_NAME), (CHAR*)protoBuffer, message_length, NX_TRUE, QOS1, NX_WAIT_FOREVER);
        	    if (ret != NX_SUCCESS) {
        	        printf("Failed to publish\n");
        	        continue;
        	    }

        	    tx_event_flags_get(&mqtt_app_flag, DEMO_ALL_EVENTS, TX_OR_CLEAR, &events, TX_WAIT_FOREVER);
        	}
#endif
        } else if (strcmp((char*) topic_buffer, MIC_REQ_TOPIC_NAME) == 0) {

        	// Code is pretty much the same as first topic, so
        	// leaving the repetitive stuff uncommented.
        	memset(topic_buffer, 0, sizeof(topic_buffer));

            uint8_t soundBuffer[iot_Sound_size];
            size_t message_length; {
                iot_Sound message = iot_Sound_init_zero;
                pb_ostream_t stream = pb_ostream_from_buffer(soundBuffer, sizeof(soundBuffer));
                uint32_t start = HAL_GetTick();
                int32_t sound = AudioRecord();
                uint32_t finish = HAL_GetTick();
                printf("Time taken for sound recording: %lu\n", (unsigned long)(finish - start));
                message.sound = sound;

                start = HAL_GetTick();
                bool status = pb_encode(&stream, iot_Sound_fields, &message);
                finish = HAL_GetTick();
                printf("Time taken for sound protobuf encoding: %lu\n", (unsigned long)(finish - start));
                message_length = stream.bytes_written;
                if (!status) {
                    printf("Error while encoding message.\n");
                    continue;
                }

                ret = nxd_mqtt_client_publish(&MqttClient, MIC_RES_TOPIC_NAME, STRLEN(MIC_RES_TOPIC_NAME), (CHAR*)soundBuffer, message_length, NX_TRUE, QOS1, NX_WAIT_FOREVER);
                if (ret != NX_SUCCESS) {
                    printf("Failed to publish\n");
                    continue;
                }

                tx_event_flags_get(&mqtt_app_flag, DEMO_ALL_EVENTS, TX_OR_CLEAR, &events, TX_WAIT_FOREVER);
            }
        } else if (strcmp((char*) topic_buffer, LIGHT_REQ_TOPIC_NAME) == 0) {

        	// Code is pretty much the same as first topic, so
        	// leaving the repetitive stuff uncommented.
        	memset(topic_buffer, 0, sizeof(topic_buffer));

        	// Initialization of light sensor + delay to make sure
        	// it's initialized properly
        	if (light_initialized == 0) {
        		uint32_t start = HAL_GetTick();
        		int ret = BSP_LIGHT_SENSOR_Init(0);
        		uint32_t finish = HAL_GetTick();
        		printf("Time taken for light initialization: %lu\n", (unsigned long)(finish - start));
				if (ret != BSP_ERROR_NONE) {
					printf("Error occured while initializing camera!\n");
					continue;
				}

                if(BSP_LIGHT_SENSOR_Start(0, LIGHT_SENSOR_MODE_CONTINUOUS) != BSP_ERROR_NONE) {
                    printf("Error while starting light sensor.\n");
                    continue;
                }

                HAL_Delay(1000);

				light_initialized = 1;
        	}

            uint8_t lightBuffer[iot_Light_size];
            size_t message_length; {
                iot_Light message = iot_Light_init_zero;
                pb_ostream_t stream = pb_ostream_from_buffer(lightBuffer, sizeof(lightBuffer));

                uint32_t start = HAL_GetTick();
                int ret = BSP_LIGHT_SENSOR_GetValues(0, pResult);
                uint32_t finish = HAL_GetTick();
                printf("Time taken for light capture: %lu\n", (unsigned long)(finish - start));

                if(ret != BSP_ERROR_NONE) {
                    printf("Error while getting light sensor values.\n");
                    continue;
                }

                message.light = pResult[0];
                /* Ensure the protobuf message is never empty — nanopb skips
                   fields with proto3 default values (0), which produces a
                   zero-length payload that NetX Duo MQTT rejects on publish. */
                if (message.light == 0) {
                    message.light = 1;
                }

                start = HAL_GetTick();
                bool status = pb_encode(&stream, iot_Light_fields, &message);
                finish = HAL_GetTick();
                printf("Time taken for light protobuf encoding: %lu\n", (unsigned long)(finish - start));
                message_length = stream.bytes_written;
                if (!status) {
                    printf("Error while encoding message.\n");
                    continue;
                }

                ret = nxd_mqtt_client_publish(&MqttClient, LIGHT_RES_TOPIC_NAME, STRLEN(LIGHT_RES_TOPIC_NAME), (CHAR*)lightBuffer, message_length, NX_TRUE, QOS1, NX_WAIT_FOREVER);
                if (ret != NX_SUCCESS) {
                    printf("Failed to publish\n");
                    continue;
                }

                tx_event_flags_get(&mqtt_app_flag, DEMO_ALL_EVENTS, TX_OR_CLEAR, &events, TX_WAIT_FOREVER);
            }
        } else if (strcmp((char*) topic_buffer, NODE_ID_REQ_TOPIC_NAME) == 0) {

        	// Code is pretty much the same as first topic, so
        	// leaving the repetitive stuff uncommented.
        	memset(topic_buffer, 0, sizeof(topic_buffer));

        	// Our node buffer is length of our node name + null terminator + additional
        	// protocol buffer overhead
            uint8_t nodeBuffer[strlen(NODE_NAME) + 1 + 2];
            size_t message_length; {
                iot_Node message = iot_Node_init_zero;
                pb_ostream_t stream = pb_ostream_from_buffer(nodeBuffer, sizeof(nodeBuffer));

                // Using the string callback here
                message.id.funcs.encode = str_write_callback;
                message.id.arg = NODE_NAME;

                uint32_t start = HAL_GetTick();
                bool status = pb_encode(&stream, iot_Node_fields, &message);
                uint32_t finish = HAL_GetTick();
                printf("Time taken for node protobuf encoding: %lu\n", (unsigned long)(finish - start));
                message_length = stream.bytes_written;
                if (!status) {
                    printf("Error while encoding message.\n");
                    continue;
                }

                ret = nxd_mqtt_client_publish(&MqttClient, NODE_ID_RES_TOPIC_NAME, STRLEN(NODE_ID_RES_TOPIC_NAME), (CHAR*)nodeBuffer, message_length, NX_TRUE, QOS1, NX_WAIT_FOREVER);
                if (ret != NX_SUCCESS) {
                    printf("Failed to publish\n");
                    continue;
                }

                tx_event_flags_get(&mqtt_app_flag, DEMO_ALL_EVENTS, TX_OR_CLEAR, &events, TX_WAIT_FOREVER);
            }
        }

        } /* end inner for(;;) */

        /* Connection lost — clean up and loop back to reconnect. */
        nxd_mqtt_client_disconnect(&MqttClient);
        tx_thread_sleep(2 * NX_IP_PERIODIC_RATE);
    } /* end outer while(1) */
}

/* USER CODE END 1 */
