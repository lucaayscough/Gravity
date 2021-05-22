#pragma once

#include <JuceHeader.h>
#include "Planet.h"


class Map: public juce::Component{
private:
    std::vector<Planet> m_Planets;
    const int M_MAX_NUM_PLANETS = 20;
    int m_NumPlanets = 0;

public:
    Map();
    ~Map() override;

    void paint(Graphics& g) override;
    void resized() override;

private:
    void createPlanet(int x, int y);
    void reservePlanetMemory();
    void mouseDoubleClick(const MouseEvent& e) override;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(Map)
};