#include "main_config.h"

void noiseResponse(SimConfig sim_config);

SimConfig sim_configs[] = {
    {
        createConcentricCores(1),
        { 1,2,3,4,5,6 },
        "r1_g2_1_3_5"
    },
    {
        createConcentricCores(1),
        { 1,2,3,4,5,6 },
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
        { 1,2,3,4,5,6 },
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