__constant sampler_t sampler = 
	  CLK_NORMALIZED_COORDS_FALSE 
	| CLK_ADDRESS_CLAMP_TO_EDGE 
	| CLK_FILTER_NEAREST;

float filterValue (__constant const float* filterWeights,
	const int x, const int y)
{
	return filterWeights[x+1 + (y+1)*(3)];
}

__kernel void filter(
    __constant int* width,
    __constant int* height,
    __constant float* filterWeightsx,
    __constant float* filterWeightsy,
    __read_only image2d_t input,
    __write_only image2d_t output)
{
	const int2 pos = {get_global_id(0), get_global_id(1)};
	float4 sumx = (float4)(0.0f);
	float4 sumy = (float4)(0.0f);
	for(int y=-1; y<=1; y++){
		for(int x=-1; x<=1; x++){
			sumx += filterValue(filterWeightsx, x, y)
			* read_imagef(input, sampler, pos + (int2)(x,y));
			sumy += filterValue(filterWeightsy, x, y)
			* read_imagef(input, sampler, pos + (int2)(x,y));
		}
	}
	float coeffx = sumx.x + sumx.y + sumx.z;
	float coeffy = sumy.x + sumy.y + sumy.z;
	write_imagef (output, (int2)(pos.x, height-pos.y), (float4)(coeffx ,coeffy,0.0f,1.0f));
}