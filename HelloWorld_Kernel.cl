__constant sampler_t sampler = 
      CLK_NORMALIZED_COORDS_FALSE 
    | CLK_ADDRESS_CLAMP_TO_EDGE 
    | CLK_FILTER_NEAREST;

typedef struct tag_kernel_nonPtr_Args{
    int width;
    int height;
    int offsetX;
    int offsetY;
    int old_offsetX;
    int old_offsetY;
}kernel_nonPtr_Args;

int to1D (int width)
{
    int x = get_local_id(0);
    int y = get_local_id(1);
    float res =  2 * (x + y * get_local_size(0));
    x = (int)res%22;
    y = floor(res/22); 
    return (x>=(22-width)/2 && y>=(22-width)/2 && x<22-(22-width)/2 && y<22-(22-width)/2)? res : -2;
}

float filterValue (__constant const float* filterWeights,
    const int x, const int y)
{
    return filterWeights[x+1 + (y+1)*(3)];
}

__kernel void filter(
    __constant kernel_nonPtr_Args* args,
    __constant float* filterWeightsBlur,
    __constant float* filterWeightsx,
    __constant float* filterWeightsy,
    __read_only image2d_t input,
    __write_only image2d_t output,
    __global float2* reductionBuffer)
{
    __local float3 local0[484];
    __local float3 local1[484];
    const int2 pos = {get_global_id(0), get_global_id(1)}; 
//BLUR
    int p = to1D(22);
    int2 initial = (int2) (get_global_id(0)-get_local_id(0),get_global_id(1)-get_local_id(1));

    for(int i=0; i<=1 && p>0; i++){
        p = p+i;
        int2 posBlur = (int2)(p%22, floor((float)p/22));
        float4 sumBlur = (float4)(0.0f);
        for(int y=-1; y<=1; y++){
            for(int x=-1; x<=1; x++){
                sumBlur += filterValue(filterWeightsBlur, x, y)
                * read_imagef(input, sampler, (int2)(initial.x-3 +posBlur.x, initial.y-3 +posBlur.y) + (int2)(x + args->offsetX, y + args->offsetY));
            }
        }
        local0 [ p ] = (float3)(sumBlur.x, sumBlur.y, sumBlur.z);  
    }
barrier(CLK_LOCAL_MEM_FENCE);

//SOBEL
    p = to1D(20);
    float2 gradient = (float2)(0.0f,0.0f);
    for(int i=0; i<=1 && p>0; i++){
        p = p+i;
        float3 sumx = (float3)(0.0f);
        float3 sumy = (float3)(0.0f);
        for(int y=-1; y<=1; y++){
            for(int x=-1; x<=1; x++){
                float3 t = local0 [ p + (x + 22*y) ];
                sumx += filterValue(filterWeightsx, x, y)
                * t;
                sumy += filterValue(filterWeightsy, x, y)
                * t;
            }
        }
        gradient = (float2)(-(sumy.x + sumy.y + sumy.z), sumx.x + sumx.y + sumx.z);
        normalize (gradient);
        local1 [ p ] = (float3)(gradient.x ,gradient.y,0.0f);      
    }
barrier(CLK_LOCAL_MEM_FENCE);

//SECOND ORDER FILTER
    p = to1D(18);
    for(int i=0; i<=1 && p>0; i++){
        p = p+i;
        float2 sumSecond = (float2)(0.0f,0.0f);
        for(int y=-1; y<=1; y++){
            for(int x=-1; x<=1 && x!=0 && y!=0; x++){
                float3 gradientNear = local1 [p + (x + 22*y)];
                gradientNear = gradientNear*(pow(gradientNear.x, 2) + pow(gradientNear.y, 2));
                sumSecond += (float2)(gradientNear.x,gradientNear.y);
            }
        }
        
        float dot = gradient.x * sumSecond.x + gradient.y * sumSecond.y;
        //dot = pow(dot, 2);
        //dot = ceil(dot-0.1f);
        gradient = (gradient + sumSecond) * dot / 2;
        local0 [ p ] = (float3)(gradient.x, gradient.y,0.0f);
        
        //float3 a = local0[p];
        //int2 posFeature = (int2)(p%22, floor((float)p/22));
        //write_imagef (output, (int2)(initial.x-3 +posFeature.x,2200-( initial.y-3 +posFeature.y)) + (int2)(args->offsetX, args->offsetY), (float4)(a.x ,a.y,0.0f,1.0f));
 
   }
barrier(CLK_LOCAL_MEM_FENCE);

//EDGE FEATURE: local sum of dot products, vector mudule equal sum of the number of vectors
    int reductionWidth = 4096+2048;
    p = to1D(16);
    for(int i=0; i<=1 && p>0; i++){
        p = p+i;
        int2 posFeature = (int2)(p%22, floor((float)p/22));
        float sumFeature = 0.0f;
        float2 sumF = (float2)(0.0f,0.0f);
        float sumDiscrete = 0.0f;
        for(int y=-1; y<=1; y++){
            for(int x=-1; x<=1; x++){
                float3 t = local0 [p + (x + 22*y)];
                float2 v = (float2) (t.x, t.y);
                float isPresent = ceil(fabs(v.x) + fabs(v.y) - 0.2f);
                sumDiscrete = sumDiscrete + isPresent;    
                sumFeature = sumFeature + ((v.x * sumF.x + v.y * sumF.y)) * isPresent;
                sumF = sumF + v * isPresent;
            }
        }
        sumFeature =  sumDiscrete - sumFeature; 
        sumF = normalize(sumF) * (sumDiscrete-1) * ceil((sumFeature - 3.0f) / 10.0f);
        
        reductionBuffer [initial.x-3 +posFeature.x + (initial.y-3 +posFeature.y)*reductionWidth] = (float2)(sumF.x, sumF.y);
        write_imagef (output, (int2)(initial.x-3 +posFeature.x, (initial.y-3 +posFeature.y)), (float4)(sumF.x,sumF.y,0.0f,1.0f));       
    }
barrier(CLK_LOCAL_MEM_FENCE);
}
__kernel void reduction(
    __constant kernel_nonPtr_Args* args,
    __write_only image2d_t output,
    __global float2* reductionBuffer)
{
    int2 id = (int2) (get_global_id(0), get_global_id(1));
    int2 expID = (int2) (id.x*2, id.y*2);
    int reductionWidth = 4096+2048;
    
    int index = expID.x + args->old_offsetX + (expID.y + args->old_offsetY) * reductionWidth;
    float2 el0 = reductionBuffer [index];
    float2 el1 = reductionBuffer [index + 1];
    float2 el2 = reductionBuffer [index + reductionWidth];
    float2 el3 = reductionBuffer [index + reductionWidth + 1];
    float2 sum =  (el0 + el1 + el2 + el3) / 4.0f;
    reductionBuffer [id.x + args->offsetX + (id.y + args->offsetY) * reductionWidth] = sum;
       
    write_imagef (output, (int2)(args->offsetX +id.x, args->offsetY +id.y), (float4)(sum.x,sum.y,0.0f,1.0f));
 
}
__kernel void matching(
    __constant kernel_nonPtr_Args* args,
    __write_only image2d_t output,
    __global float2* reductionBuffer0,
    __global float2* reductionBuffer1)
{
    int2 id = {get_global_id(0)*2,get_global_id(1)*2};
    int2 myPos = {id.x + args->offsetX, id.y + args->offsetY};
    int reductionWidth = 4096+2048;
    
   //get offset computed before
   int2 startPosition;
   if(args-> old_offsetX < 0){
       startPosition = (int2)(id.x + args->offsetX, id.y + args->offsetY);
   }else{
       float2 oldPos = reductionBuffer0[args->old_offsetX + id.x/2 + (args->old_offsetY + id.y/2)* reductionWidth] * 2;
       startPosition = (int2)((int)(oldPos.x) + args->offsetX, (int)(oldPos.y) + args->offsetY);
   }
       
   //compute the 4 positions
   for(int x=0; x<2; x++){
       for(int y=0; y<2; y++){
           float2 myValue = reductionBuffer0[(x + myPos.x) + (y + myPos.y) * reductionWidth];
           float minDiff = INFINITY;
           int minCoordX = 0, minCoordY = 0;
           for(int yLoc=-1; yLoc<=1; yLoc++){
               for(int xLoc=-1; xLoc<=1; xLoc++){
                   float2 diff2 = myValue - reductionBuffer1[xLoc + x + startPosition.x + (yLoc + y + startPosition.y)*reductionWidth];
                   float diff = fast_length(diff2);
                   if(diff < minDiff){
                        minDiff = diff;
                        minCoordX = x;
                        minCoordY = y;
                   }
               }
           }
           float2 newCoord = (float2)((float)(minCoordX + x + startPosition.x - args->offsetX), (float)(minCoordY + y + startPosition.y - args->offsetY));
      
           reductionBuffer0[(x + myPos.x) + (y + myPos.y) * reductionWidth] = newCoord;
           write_imagef (output, (int2)((x + myPos.x) , (y + myPos.y)), (float4)(newCoord.x/1024.0f,newCoord.y/512.0f,0.0f,1.0f));        
       }
   } 
}