#ifndef DATA_BLOB_H
#define DATA_BLOB_H
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif
typedef struct ts_blob {
   uint8_t *data;
   uint32_t size;
   uint8_t *meta;
   uint32_t meta_size;
   int32_t id;
   int8_t free_data;
} blob;

blob* new_blob_and_copy_data(int32_t id, uint32_t size, uint8_t* data);
blob* new_blob_and_move_data(int32_t id, uint32_t size, uint8_t* data);
blob* new_blob_and_alloc_data(int32_t id, uint32_t size);
blob* new_empty_blob(int32_t id);
void free_blob(blob* temp);

void fill_blob_meta(blob* temp, uint32_t meta_size, uint8_t* meta);
void copy_blob_meta(blob* dest, blob* src);

blob* write_file_to_blob(const char *filename);
void write_blob_to_file(const char *filename, blob* temp);
#ifdef __cplusplus
}
#endif

#endif
