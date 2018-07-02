#include "data_blob.h"
#include <stdio.h>
blob* new_blob_and_copy_data(int32_t id, uint32_t size, uint8_t* data)
{
   blob *temp = (blob*)malloc(sizeof(blob));
   temp->data = (uint8_t*)malloc(sizeof(uint8_t)*size);
   temp->meta = NULL;
   temp->meta_size = 0;
   memcpy(temp->data, data, size);
   temp->size = size;
   temp->id = id;
   temp->free_data = 1;
   return temp; 
}

blob* new_blob_and_move_data(int32_t id, uint32_t size, uint8_t* data)
{
   blob *temp = (blob*)malloc(sizeof(blob));
   temp->data = data;
   temp->size = size;
   temp->meta = NULL;
   temp->meta_size = 0;
   temp->id = id;
   temp->free_data = 0;
   return temp; 
}

blob* new_blob_and_alloc_data(int32_t id, uint32_t size)
{
   blob *temp = (blob*)malloc(sizeof(blob));
   temp->data = (uint8_t*)malloc(sizeof(uint8_t)*size);
   temp->size = size;
   temp->meta = NULL;
   temp->meta_size = 0;
   temp->id = id;
   temp->free_data = 1;
   return temp; 
}

blob* new_empty_blob(int32_t id)
{
   blob *temp = (blob*)malloc(sizeof(blob));
   temp->data = NULL;
   temp->size = 0;
   temp->meta = NULL;
   temp->meta_size = 0;
   temp->id = id;
   temp->free_data = 0;
   return temp; 
}

void free_blob(blob* temp){
   if(temp->meta_size > 0){
      free(temp->meta);
   }
   if((temp->data != NULL)&&(temp->free_data==1)){
      free(temp->data);
   }
   free(temp);
}

blob* write_file_to_blob(const char *filename){
   uint32_t size = 0;
   blob* temp;
   FILE *f = fopen(filename, "rb");
   if (f == NULL){
      return NULL;
   }
   fseek(f, 0, SEEK_END);
   size = ftell(f);
   fseek(f, 0, SEEK_SET);
   temp = new_blob_and_alloc_data(0, size+1);
   if(temp == NULL) return NULL;
   if (size != fread(temp->data, sizeof(char), size, f)){
      free_blob(temp);
      return NULL;
   }
   fclose(f);
   ((uint8_t*)temp->data)[size] = 0;
   temp->size = size;
   return temp;
}

void write_blob_to_file(const char *filename, blob* temp){
   FILE *fp;
   fp = fopen( filename , "wb" );
   fwrite(temp->data, 1,  (temp->size)*sizeof(uint8_t), fp );
   fclose(fp);
}

void fill_blob_meta(blob* temp, uint32_t meta_size, uint8_t* meta){
   if(temp->meta_size > 0) free(temp->meta);
   temp->meta = (uint8_t*)malloc(sizeof(uint8_t)*meta_size);
   temp->meta_size = meta_size;
   memcpy(temp->meta, meta, meta_size);
}

void copy_blob_meta(blob* dest, blob* src){
   dest->meta = (uint8_t*)malloc(sizeof(uint8_t)*src->meta_size);
   dest->meta_size = src->meta_size;
   memcpy(dest->meta, src->meta, src->meta_size);
}


