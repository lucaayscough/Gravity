#pragma once


class Planet: public juce::Component{
public:
    bool m_Destroy = false;

private:
    juce::ComponentDragger m_Dragger;

    const int M_SIZE_MODIFIER = 2;
    const int M_MAX_PLANET_SIZE = 100;
    const int M_MIN_PLANET_SIZE = 20;

    int m_Diameter = 50;
    int m_WindowWidth;
    int m_WindowHeight;
    
    // Used to update the state in the map object
    // when a planet is set to be destroyed.
    juce::Value* m_DestroyPlanetPtr;

    // Used to access the generator instantiated in
    // the PluginProcessor.
    Generator* m_GeneratorPtr;

    // Generated sound.
    at::Tensor m_Latents;
    juce::Array<float> m_Sample;


public:
    Planet();
    Planet(juce::Value*, Generator*);
    ~Planet() override;

    void paint(Graphics&) override;
    void resized() override;
    
    void draw(int, int, int);
    void resizePlanet(int);

    void setDiameter(int);
    void setMapBoundaries(int, int);

    int getDiameter();

    void generateSample();

private:
    void mouseDown(const MouseEvent&) override;
    void mouseDrag(const MouseEvent&) override;
    void mouseWheelMove(const MouseEvent&, const MouseWheelDetails&) override;
    void visibilityChanged() override;
    
    void checkBounds();

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(Planet)
};