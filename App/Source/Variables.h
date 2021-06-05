#pragma once


struct Variables{
    // Window member variables.
    static const int WINDOW_WIDTH;
    static const int WINDOW_HEIGHT;
    static const bool IS_WIDTH_RESIZABLE;
    static const bool IS_HEIGHT_RESIZABLE;

    // Map member variables.
    static const int MAX_NUM_PLANETS;
    
    // Planet member variables.
    static const int DEFAULT_PLANET_DIAMETER;
    static const int MAX_PLANET_SIZE;
    static const int MIN_PLANET_SIZE;
    static const int SIZE_MODIFIER;
    static const int CLIP_BOUNDARY;

    // Sun member variables.
    static const int SUN_DIAMETER;
};