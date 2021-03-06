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
    int distance;
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
        gradient /= sqrt(32.0f);   //ensure that gradient is in [-1.0,1.0] length range
        local1 [ p ] = (float3)(gradient.x ,gradient.y,0.0f);      
        
        /*int reductionWidth = 4096+2048;
        float2 a = gradient/1.0f;
        int2 posFeature = (int2)(p%22, floor((float)p/22));
        //reductionBuffer [initial.x-3 +posFeature.x + (initial.y-3 +posFeature.y)*reductionWidth] = (float2)(a.x, a.y);
    write_imagef (output, (int2)(initial.x-3 +posFeature.x,2200-( initial.y-3 +posFeature.y)) + (int2)(args->offsetX, args->offsetY), (float4)(a.x ,a.y,0.0f,1.0f));*/
 
    }
barrier(CLK_LOCAL_MEM_FENCE);

//SECOND ORDER FILTER
    p = to1D(18);   ///TO CHANGE    <---
    for(int i=0; i<=1 && p>0; i++){
        p = p+i;
        float2 sumSecond = (float2)(0.0f,0.0f);
        for(int y=-1; y<=1; y++){
            for(int x=-1; x<=1 && !(x==0 && y==0); x++){        
                float3 gradientNear = local1 [p + (x + 22*y)];
                gradientNear = gradientNear*(pow(gradientNear.x, 2) + pow(gradientNear.y, 2));
                sumSecond += (float2)(gradientNear.x,gradientNear.y);
            }
        }
        sumSecond = normalize(sumSecond);
        float Dot = dot(gradient,sumSecond);
        gradient = (gradient + sumSecond) * Dot / 2;
        local0 [ p ] = (float3)(gradient.x, gradient.y,0.0f);
        
        /*int reductionWidth = 4096+2048;
        float2 a = gradient*10.0f;
        int2 posFeature = (int2)(p%22, floor((float)p/22));
        //reductionBuffer [initial.x-3 +posFeature.x + (initial.y-3 +posFeature.y)*reductionWidth] = (float2)(a.x, a.y);
    write_imagef (output, (int2)(initial.x-3 +posFeature.x,2200-( initial.y-3 +posFeature.y)) + (int2)(args->offsetX, args->offsetY), (float4)(a.x ,a.y,0.0f,1.0f));*/
 
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
                float isPresent = ceil(fabs(v.x) + fabs(v.y) - 0.1f);
                sumDiscrete = sumDiscrete + isPresent;    
                sumFeature = sumFeature + ((v.x * sumF.x + v.y * sumF.y)) * isPresent;
                sumF = sumF + v * isPresent;
            }
        }
        sumFeature =  sumDiscrete - sumFeature; 
        sumF = normalize(sumF) * (sumDiscrete-1) * ceil((sumFeature - 3.0f) / 10.0f);
        
        reductionBuffer [initial.x-3 +posFeature.x + (initial.y-3 +posFeature.y)*reductionWidth] = (float2)(sumF.x, sumF.y);
        write_imagef (output, (int2)(initial.x-3 +posFeature.x, (initial.y-3 +posFeature.y)), (float4)(sumF.x*5.0f,sumF.y*5.0f,0.0f,1.0f));     
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
    float2 sum =  (el0 + el1 + el2 + el3);
    reductionBuffer [id.x + args->offsetX + (id.y + args->offsetY) * reductionWidth] = sum;
       
    write_imagef (output, (int2)(args->offsetX +id.x, args->offsetY +id.y), (float4)(sum.x,sum.y,0.0f,1.0f));
    
barrier(CLK_GLOBAL_MEM_FENCE);
}
__kernel void matching(
    __constant kernel_nonPtr_Args* args,
    __write_only image2d_t output,
    __global float2* reductionBuffer0,
    __global float2* reductionBuffer1)
{
    int2 id = {get_global_id(0)*2,get_global_id(1)*2};
    int2 dim = {get_global_size(0)*2,get_global_size(1)*2};
    int2 myPos = {id.x + args->offsetX, id.y + args->offsetY};
    int reductionWidth = 4096+2048;
    
   //get offset computed before
   int2 startPosition;
   if(args-> old_offsetY < 0){
       startPosition = myPos;
   }else{
       float2 oldPos = reductionBuffer0[args->old_offsetX + id.x/2 + (args->old_offsetY + id.y/2)* reductionWidth] *2;
       startPosition = (int2)((int)(oldPos.x) + args->offsetX, (int)(oldPos.y) + args->offsetY);
   }
       
   //compute the 4 positions
   for(int x=0; x<2; x++){
       for(int y=0; y<2; y++){
       //1) get central value
          float myValue = 0.0f;
           for(int yLoc=-1; yLoc<=1; yLoc++){
               for(int xLoc=-1; xLoc<=1; xLoc++){
                   if(id.x + x + xLoc < dim.x && id.x + x + xLoc >= 0 && id.y + y + yLoc< dim.y && id.y + y + yLoc >= 0 )
                   //if(1)
                        myValue += (1/9) * dot(reductionBuffer0[(xLoc + x + startPosition.x) + (yLoc + y + startPosition.y) * reductionWidth], (float2)(0.7071f,0.7071f));
                        else myValue += (1/9) * dot(reductionBuffer0[( x + startPosition.x) + ( y + startPosition.y) * reductionWidth], (float2)(0.7071f,0.7071f));
               }
           }
       //2)get weightened sum for each other value and subtract
        __local float global1[5][5];
        __local float local1[3][3];
        for(int yLoc=-1; yLoc<=1; yLoc++){
            for(int xLoc=-1; xLoc<=1; xLoc++){   
                    float otherValue = 0.0f;                
                    for(int ySub=-1; ySub<=1; ySub++){
                        for(int xSub=-1; xSub<=1; xSub++){
                             //if(id.x + x + xLoc + xSub < dim.x && id.x + x + xLoc + xSub >= 0 && id.y + y + yLoc + ySub < dim.y && id.y + y + yLoc + ySub >= 0 ){
                                if(1){
                                float d = dot(reductionBuffer1[(xSub + xLoc + x + startPosition.x) + (ySub + yLoc + y + startPosition.y) * reductionWidth], (float2)(0.7071f,0.7071f));
                                otherValue += (1/9) * d;
                                global1[xSub + xLoc + 2][ySub + yLoc + 2] = d;  
                            }         
                            else otherValue += (1/9) * dot(reductionBuffer1[(xLoc + x + startPosition.x) + ( yLoc + y + startPosition.y) * reductionWidth], (float2)(0.7071f,0.7071f));
                        }
                    }
                    if(id.x + x + xLoc < dim.x && id.x + x + xLoc >= 0 && id.y + y + yLoc< dim.y && id.y + y + yLoc >= 0 )
                        local1[xLoc+1][yLoc+1] = otherValue - myValue;
                    else local1[xLoc+1][yLoc+1] = 0;
               }
           }
        //3)determination of points in each 4 subpixel
        __local float minValues[2][2];
        __local float2 minPositions[2][2];
            for(int grX=0; grX<2; grX++){
                for(int grY=0; grY<2; grY++){  
                    float val0 = local1[grX][grY];
                    float val1 = local1[grX+1][grY];
                    float val2 = local1[grX][grY+1];
                    float val3 = local1[grX+1][grY+1];
                    float dist0 = val1 - val0;  float dist1 = val3 - val1;
                    float dist2 = val3 - val2;  float dist3 = val2 - val0;
                    float2 vertices[4]; 
                    if(dist0 > 0) vertices[0] = (float2)(-val0 / fabs(dist0), 0.0f);
                        else vertices[0] = (float2)((val1) / fabs(dist0) + 1, 0.0f);
                    if(dist1 > 0) vertices[1] = (float2)(0.0f, -val1 / fabs(dist1));
                        else vertices[1] = (float2)(0.0f, (val3) / fabs(dist1) + 1);
                    if(dist2 > 0) vertices[2] =(float2)( -val2 / fabs(dist2), 0.0f);
                        else vertices[2] = (float2)((val3) / fabs(dist2) + 1, 0.0f);
                    if(dist3 > 0) vertices[3] = (float2)(0.0f, -val0 / fabs(dist3));
                        else vertices[3] = (float2)(0.0f, (val2) / fabs(dist3) + 1);
                        
                    //find the relative local minimum
                    float2 minPos = (vertices[0] + vertices[1] + vertices[2] + vertices[3]) / 4.0f;
                    //offbound values
                    minPos = (float2)(clamp(minPos.x, -0.5f, 1.5f), clamp(minPos.y, -0.5f, 1.5f));
                    //save min local->global pos
                    minPos = minPos + (float2)((float)grX,(float)grY); //9x9 origin on top left
                    
                    //interpolation
                    int posLeft = floor(minPos.x); int posTop = floor(minPos.y);
                    float interpX0= (1 - minPos.x - posLeft) * global1[posLeft+1][posTop+1] + (minPos.x - posLeft) * global1[posLeft+2][posTop+1];
                    float interpX1= (1 - minPos.x - posLeft) * global1[posLeft+1][posTop+2] + (minPos.x - posLeft) * global1[posLeft+2][posTop+2];
           
                    minValues[grX][grY] = (1 - minPos.y - posTop) * interpX0 + (minPos.y - posTop) * interpX1;
                    minPositions[grX][grY] = minPos + (float2)(0.5f,0.5f) - (float2)(1.5f, 1.5f); //9x9 origin on center
                } 
            }
            //find local minimum
            float minDiff = INFINITY; float2 minP = (float2)(0.0f,0.0f);
            for(int Sx = 0;Sx<2;Sx++){
                for(int Sy = 0;Sy<2;Sy++){
                    if(fabs(minValues[Sx][Sy]-myValue)<minDiff){
                        minDiff = fabs(minValues[Sx][Sy]-myValue);
                        minP = minPositions[Sx][Sy];
                    }
                }
            }
            float2 newCoord = (float2)((float)(minP.x + x + startPosition.x - args->offsetX), (float)(minP.y + y + startPosition.y - args->offsetY));
      
            reductionBuffer0[(x + myPos.x) + (y + myPos.y) * reductionWidth] = newCoord;
           write_imagef (output, (int2)((x + myPos.x) , (y + myPos.y)), (float4)(newCoord.x/(float)get_global_size(0)/2,newCoord.y/(float)get_global_size(1)/2,0.0f,1.0f));        
       }
   } 
barrier(CLK_GLOBAL_MEM_FENCE);
}
__kernel void triangulate(
    __constant kernel_nonPtr_Args* args,
    __global float4* vertexBuffer,
    __global float2* reductionBuffer,
    __write_only image2d_t output)
{
    int2 id = {get_global_id(0),get_global_id(1)};
    int2 dim = {get_global_size(0), get_global_size(1)};
    int2 myPos = {id.x + args->offsetX, dim.y-id.y + args->offsetY};
    int reductionWidth = 4096+2048;
    
    float2 alpha0 = (float2){ M_PI_2_F -  M_PI_F * (id.x - dim.x/2) / 1664, M_PI_2_F * (id.y - dim.y/2) / 832};
    float2 alpha1 = reductionBuffer[(myPos.x) + (myPos.y) * reductionWidth];
    alpha1 = (float2){ M_PI_2_F - M_PI_F * ((int)alpha1.x - dim.x/2) / 1664, M_PI_2_F * ((int)(dim.y-alpha1.y) - dim.y/2) / 832};
    
    float a0 = cos(alpha0.x) * cos(alpha0.y);   float a1 = cos(alpha1.x) * cos(alpha1.y);
    float b0 = sin(alpha0.x) * cos(alpha0.y);   float b1 = sin(alpha1.x) * cos(alpha1.y);
    float c0 = sin(alpha0.y);                   float c1 = sin(alpha1.y);
    float3 v0 = (float3)(a0,b0,c0);
    float3 v1 = (float3)(a1,b1,c1);
    float3 q0 = (float3)(0,0,0);
    float3 q1 = (float3)(0,-args->distance,0);
    
    //get point with minimum distance from both lines
    float3 n = normalize(cross(v0, v1));
    float d = fabs(dot(n, (q1-q0)));
    float3 n1 = cross(v0, n);
    float3 n2 = cross(v1, n);
    
    float3 p0 = q0 + (dot((q1-q0), n2)/dot(v0,n2))*v0;
    float3 p1 = q1 + (dot((q0-q1), n1)/dot(v1,n1))*v1;
    
    float3 point = (p0+p1) / 2.0f;
    
    float len = length(point);
    if(len> 30.0f || isnan(point).x<0) point = v0 * 30.0f;    //if the point is too far away project onto a sphere

    vertexBuffer[id.x + id.y * dim.x] = (float4)(-point.x, point.z, point.y,1.0f);
 
barrier(CLK_GLOBAL_MEM_FENCE); 
}