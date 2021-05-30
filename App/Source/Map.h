#pragma once


class Map: public juce::Component, public juce::Value::Listener{
private:
    // Sun container.
    juce::OwnedArray<Sun> m_Sun;

    // Planet container and variables.
    OwnedArray<Planet> m_Planets;
    const int M_MAX_NUM_PLANETS = 20;
    int m_NumPlanets = 0;

    Generator* m_GeneratorPtr;

public:
    Map();
    Map(Generator*);
    ~Map() override;

    void paint(Graphics&) override;
    void resized() override;
    void createSun();

private:
    void createPlanet(int, int);
    void setPlanetID(Planet*);
    void setupPlanet(Planet*, int x, int y);
    void destroyPlanet();

    void mouseDoubleClick(const MouseEvent&) override;
    void valueChanged(juce::Value&) override;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(Map)
};