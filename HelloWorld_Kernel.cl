__constant sampler_t sampler = 
      CLK_NORMALIZED_COORDS_FALSE 
    | CLK_ADDRESS_CLAMP_TO_EDGE 
    | CLK_FILTER_NEAREST;

typedef struct tag_kernel_nonPtr_Args{
    int width;
    int height;
    int offsetX;
    int offsetY;
}kernel_nonPtr_Args;

int to1D (int width){
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
    __write_only image2d_t output)
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
        for(int y=-1; y<=1 && y!=0; y++){
            for(int x=-1; x<=1 && x!=0; x++){
                float3 gradientNear = local1 [p + (x + 22*y)];
                gradientNear = gradientNear*(pow(gradientNear.x, 2) + pow(gradientNear.y, 2));
                sumSecond += (float2)(gradientNear.x,gradientNear.y);
            }
        }
        normalize(sumSecond);
        float dot = gradient.x * sumSecond.x + gradient.y * sumSecond.y;
        dot = pow(dot, 2);
        dot = ceil(dot-0.1f);
        gradient = gradient * dot;
        local0 [ p ] = (float3)(gradient.x, gradient.y,0.0f);
   }
barrier(CLK_LOCAL_MEM_FENCE);

//EDGE FEATURE: local sum of dot products, vector mudule equal sum of the number of vectors
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
            float isPresent = ceil(fabs(v.x) + fabs(v.y) - 0.1f);
            sumDiscrete = sumDiscrete + isPresent;    
            sumFeature = sumFeature + (1 - (v.x * sumF.x + v.x * sumF.y)) * isPresent;
            sumF = sumF + v;
            }
        }
        sumF = normalize(sumF) * (sumDiscrete-1) * ceil((sumFeature - 6.0f) / 10.0f);
        write_imagef (output, (int2)(initial.x-3 +posFeature.x,2000-( initial.y-3 +posFeature.y)) + (int2)(args->offsetX, args->offsetY), (float4)(sumF.x ,sumF.y,0.0f,1.0f));
    }
barrier(CLK_LOCAL_MEM_FENCE);
}