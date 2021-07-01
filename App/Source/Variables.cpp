#include "Headers.h"


// Window variables.
const int Variables::WINDOW_WIDTH = 1080;
const int Variables::WINDOW_HEIGHT = 720;
const bool Variables::IS_WIDTH_RESIZABLE = false;
const bool Variables::IS_HEIGHT_RESIZABLE = false;

// Editor variables.
const juce::Colour Variables::EDITOR_BG_COLOUR = juce::Colour(46, 48, 53);
const int Variables::LEFT_BAR = 180;
const int Variables::TOP_BAR = 40;
const int Variables::MAP_TRIM = 8;

// Top bar variables.
const juce::Colour Variables::TOP_BAR_SHADOW_COLOUR = juce::Colour(16, 16, 16);

// Map variables.
const int Variables::NUM_MAPS = 8;
const int Variables::MAX_NUM_PLANETS = 20;
const juce::Colour Variables::MAP_BG_COLOUR_1 = juce::Colour(37, 38, 43);
const juce::Colour Variables::MAP_BG_COLOUR_2 = juce::Colour(33, 34, 38);
const juce::Colour Variables::MAP_CIRCLE_COLOUR = juce::Colour(86, 87, 90);
const float Variables::FORCE_VECTOR_SIZE = 1.0f;

// Planet variables.
const float Variables::DEFAULT_PLANET_AREA = 2000.0f;
const float Variables::MAX_PLANET_AREA = 5000.0f;
const float Variables::MIN_PLANET_AREA = 1000.0f;
const float Variables::AREA_MODIFIER = 200.0f;
const int Variables::CLIP_BOUNDARY = 100.0f;

// Sun variables.
const float Variables::SUN_AREA = 7000.0f;
