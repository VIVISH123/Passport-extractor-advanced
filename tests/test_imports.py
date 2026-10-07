def test_imports():
    import extractor
    import preprocess
    import mrz_parser
    import validator
    import visual_ocr
    import stage2

    assert callable(extractor.extract_passport)
    assert callable(preprocess.preprocess_image)
    assert callable(mrz_parser.parse_mrz)
    assert callable(validator.build_result)
    assert callable(visual_ocr.extract_visual_fields)
    assert callable(stage2.run_stage2)
