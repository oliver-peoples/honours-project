#include "main_config.h"

void noiseResponse(SimConfig sim_config);

SimConfig sim_configs[] = {
    {
        createConcentricCores(2),
        { 7,11,15 },
        "r2_g2_7_11_15"
    },
    {
        createConcentricCores(2),
        { 0,7,11,15 },
        "r2_g2_0_7_11_15"
    },
    {
        createConcentricCores(2),
        { 7,8,9,10,11,12,13,14,15,16,17,18 },
        "r2_g2_7_8_9_10_11_12_13_14_15_16_17_18"
    },
    {
        createConcentricCores(2),
        { 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18 },
        "r2_g2_all"
    },
    {
        createConcentricCores(1),
        { 1,3,5 },
        "r1_g2_1_3_5"
    },
    {
        createConcentricCores(1),
        { 0,1,3,5 },
        "r1_g2_0_1_3_5"
    },
    {
        createConcentricCores(1),
        { 1,2,3,4,5,6 },
        "r1_g2_1_2_3_4_5_6"
    },
    {
        createConcentricCores(2),
        { 1,2,3,4,5,6 },
        "r2_g2_1_2_3_4_5_6"
    },
    {
        createConcentricCores(2),
        { 1,3,5 },
        "r2_g2_1_3_5"
    }
};

int main()
{
    for (SimConfig sim_config : sim_configs)
    {
        noiseResponse(sim_config);
    }
    
    return 0;
}