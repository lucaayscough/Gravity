#pragma once


class MapButton: public juce::Component{
public:
    // Constructors and destructors.
    MapButton();
    ~MapButton() override;

public:
    // View methods.
    void paint(Graphics&) override;
    void resized() override;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(MapButton)
};
