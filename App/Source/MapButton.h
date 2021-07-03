#pragma once


class MapButton: public juce::Component{
public:
    // Constructors and destructors.
    MapButton(juce::OwnedArray<Map>&);
    ~MapButton() override;

private:
    // View methods.
    void paint(Graphics&) override;
    void resized() override;

public:
    // Interface methods.
    int getButtonIndex();
    Map& getMap();

private:
    // Controller methods.
    void mouseDown(const MouseEvent&) override;

private:
    // Member variables.
    juce::OwnedArray<Map>& m_MapsRef;
    juce::ImageComponent m_MapImage;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(MapButton)
};
