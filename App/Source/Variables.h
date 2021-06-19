#pragma once


struct Variables{
    // Window member variables.
    static const int WINDOW_WIDTH;
    static const int WINDOW_HEIGHT;
    static const bool IS_WIDTH_RESIZABLE;
    static const bool IS_HEIGHT_RESIZABLE;

    // Main component sizes.
    static const int LEFT_BAR;
    static const int TOP_BAR;
    static const int MAP_TRIM;

    // Map member variables.
    static const int MAX_NUM_PLANETS;
    static const juce::Colour MAP_BG_COLOUR_1;
    static const juce::Colour MAP_BG_COLOUR_2;
    static const juce::Colour MAP_CIRCLE_COLOUR;
    
    // Planet member variables.
    static const int DEFAULT_PLANET_DIAMETER;
    static const int MAX_PLANET_SIZE;
    static const int MIN_PLANET_SIZE;
    static const int SIZE_MODIFIER;
    static const int CLIP_BOUNDARY;

    // Sun member variables.
    static const int SUN_DIAMETER;
};