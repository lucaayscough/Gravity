#pragma once


class Map: public juce::Component, public juce::Value::Listener{
private:
    // Sun container.
    OwnedArray<Sun> m_Sun;

    // Planet container and variables.
    OwnedArray<Planet> m_Planets;
    const int M_MAX_NUM_PLANETS = 20;
    int m_NumPlanets = 0;

    // This value gets updated by the planet class
    // when it is set to be destroyed.
    juce::Value m_DestroyPlanet;

    Generator* m_GeneratorPtr;

public:
    Map();
    ~Map() override;

    void paint(Graphics&) override;
    void resized() override;
    void createSun();

    void setGeneratorAccess(Generator*);

private:
    void createPlanet(int, int);
    void destroyPlanet();

    void mouseDoubleClick(const MouseEvent&) override;
    void valueChanged(juce::Value&) override;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(Map)
};