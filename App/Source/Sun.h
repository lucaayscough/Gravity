#pragma once


class Sun : public Planet{
private:
    const int M_DIAMETER = 75;

    // Used to access the generator instantiated in
    // the PluginProcessor.
    Generator* m_GeneratorPtr;

public:
    Sun();
    Sun(Generator*);
    ~Sun() override;

    void paint(Graphics& g) override;
    void resized() override;

    int getDiameter();
    
    void generateLatents();
    void generateSample();

private:
    void mouseDown(const MouseEvent& e) override;
    void mouseDrag(const MouseEvent& e) override;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(Sun)
};