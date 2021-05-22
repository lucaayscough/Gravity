#pragma once

#include <JuceHeader.h>
#include "Planet.h"


class Map: public juce::Component{
private:
    std::vector<Planet> m_Planets;

public:
    Map();
    ~Map() override;
    
    void paint(Graphics& g) override;
    void createPlanet(int x, int y);

private:
    void mouseDoubleClick(const MouseEvent& e) override;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(Map)
};