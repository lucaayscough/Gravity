#pragma once


class LeftBar: public juce::Component{
public:
    // Constructors and destructors.
    LeftBar();
    ~LeftBar() override;

public:
    // View methods.
    void paint(Graphics&) override;
    void resized() override;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(LeftBar)
};
