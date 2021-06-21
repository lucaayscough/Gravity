#pragma once


// TODO:
// Cleanup this header file.
// Sort private and public access.

class Planet: public Astro{
public: 
    // Constructors and destructors.
    Planet(juce::OwnedArray<Planet>&, AudioContainer&, Parameters&, ControlPanel&);
    ~Planet() override;
   
public:
    // View methods.
    void paint(Graphics&) override;
    void resized() override;
    void draw() override;
    void draw(int, int, int) override;
    void resizePlanet(int);

public:
    // Interface methods.
    void setCentrePosXY(int, int) override;

    juce::ValueTree getState() override;
    int getClipBoundary();

    void checkCollision();
    void checkBounds();

private:
    // Controller methods.
    bool hitTest(int, int) override;
    void mouseEnter(const MouseEvent&) override;
    void mouseExit(const MouseEvent&) override;
    void mouseDown(const MouseEvent&) override;
    void mouseUp(const MouseEvent&) override;
    void mouseDrag(const MouseEvent&) override;
    void mouseWheelMove(const MouseEvent&, const MouseWheelDetails&) override;

private:
    // Callback methods.
    void visibilityChanged() override;
    void valueChanged(juce::Value&) override;

private:
    // Member variables.
    juce::ComponentDragger m_Dragger;
    juce::OwnedArray<Planet>& m_PlanetsRef;
    juce::ColourGradient m_ColourGradient;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(Planet)
};


