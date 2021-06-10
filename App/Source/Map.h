#pragma once


class Map: public juce::Component, public juce::Value::Listener{
public:
    // Constructors and destructors.
    Map(AudioContainer&, Parameters&);
    ~Map() override;

private:
    // View methods.
    void paint(Graphics&) override;
    void resized() override;
    
    void createSun();
    void createPlanet(int, int);
    void setupPlanet(Planet*, int x, int y, juce::ValueTree);
    void destroyPlanet();
    void rebuildPlanets();

    // Interface methods
    int getMaxNumPlanets();
    int getNumPlanets();
    float getDistance(Sun&, Planet*);
    float getDistance(Planet*, Planet*);

    // Controller methods.
    void mouseUp(const MouseEvent&) override;
    void mouseDoubleClick(const MouseEvent&) override;
    void valueChanged(juce::Value&) override;

    OwnedArray<Planet> m_Planets;
    AudioContainer& m_AudioContainerRef;
    Parameters& m_ParametersRef;
    Sun m_Sun;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(Map)
};