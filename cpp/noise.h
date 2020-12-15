/** @file noise.h
 *  @brief For easy sampling of noise from distributions in C++.
 */

#ifndef D3T_PRIVACYNOISE_H
#define D3T_PRIVACYNOISE_H
#include "utils.h"
#include <random>

class Noise {
public:
    Noise(int seed, bool turnOffNoise) : rng_(seed), turnOffNoise_(turnOffNoise)
    {
        if (turnOffNoise_) {
            std::cout << "Turned off noise!" << std::endl;
        }
        else {
            std::cout << "Noise with seed " << seed << std::endl;
        }
    }

    float laplace(float b)
    {
        if (turnOffNoise_) {
            return 0.0;
        }
        else {
            std::exponential_distribution<float> expDist(1.0 / b);
            float result = expDist(rng_) - expDist(rng_);
            return result;
        }
    }

private:
    std::mt19937 rng_;
    bool turnOffNoise_;
};

#endif // D3T_PRIVACYNOISE_H
