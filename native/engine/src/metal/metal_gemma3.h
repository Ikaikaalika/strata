// Metal GPU backend for Gemma 3 (implemented in metal_gemma3.mm).
#pragma once

#include <memory>
#include <string>

#include "gemma3.h"
#include "model.h"

namespace lokahi {

bool metal_available();
std::string metal_device_name();

std::unique_ptr<Model> make_gemma3_metal(std::shared_ptr<Gemma3Weights> weights,
                                         const LoadOptions& options);

}  // namespace lokahi
