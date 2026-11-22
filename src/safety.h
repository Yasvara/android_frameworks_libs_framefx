/*
 * SPDX-FileCopyrightText: IRedDragonICY
 * SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0
 */

#pragma once

// Auto-disable for the risky features: each leaves a marker in the game's cache before it first
// runs, cleared once the game has presented a while with it. A marker still there at the next
// start means the game died with the feature on: that start runs without it, and the one after
// tries again. The log says which feature and why.
namespace framefx::safety {

enum Feature { kShadingRate = 0, kRays = 1, kTransparencyCut = 2, kFeatureCount };

// Whether `feature` may run this start; arms its marker when it may. Asked once per start.
bool allowed(Feature feature);

// One present of the game's: markers clear after kCleanPresents.
void onPresent();

}  // namespace framefx::safety
