import pytest
from unittest.mock import MagicMock, AsyncMock, patch

class MockPreprocessedQuery:
    def __init__(self):
        self.vector_query = [0.1] * 1536
        self.use_hybrid = False
        self.has_time = False
        self.time_confidence = 0.0
        self.hybrid_reason = "none"
        self.keyword = ""
        self.stripped_query = "test"

class MockHit:
    def __init__(self, id, score):
        self.id = id
        self.score = float(score)
        self.payload = {}

@pytest.mark.asyncio
async def test_bundle_search_no_results():
    """
    Test when vector search returns no results.
    This validates the basic entry point and the short-circuit logic.
    """
    from m_flow.retrieval.episodic.bundle_search import episodic_bundle_search
    from m_flow.retrieval.episodic.config import EpisodicConfig
    
    mock_config = EpisodicConfig()
    mock_prep_query = MockPreprocessedQuery()
    
    with patch("m_flow.retrieval.episodic.bundle_search.preprocess_query", return_value=mock_prep_query), \
         patch("m_flow.retrieval.episodic.bundle_search._vector_search", new_callable=AsyncMock) as mock_vec:
        
        # Mock vector search to return empty hits
        mock_vec.return_value = ({}, {})
        
        results = await episodic_bundle_search("empty query", config=mock_config)
        
        assert results == []
        mock_vec.assert_called_once()

@pytest.mark.asyncio
async def test_bundle_search_with_mocked_bundles():
    """
    Test the flow when bundles are found but no edges are returned.
    This validates the pipeline up to the output assembly stage.
    """
    from m_flow.retrieval.episodic.bundle_search import episodic_bundle_search
    from m_flow.retrieval.episodic.config import EpisodicConfig
    
    mock_config = EpisodicConfig()
    mock_prep_query = MockPreprocessedQuery()
    
    # Create a fragment mock that has nodes and edges
    mock_fragment = MagicMock()
    mock_fragment.nodes = {}
    mock_fragment.edges = {}

    with patch("m_flow.retrieval.episodic.bundle_search.preprocess_query", return_value=mock_prep_query), \
         patch("m_flow.retrieval.episodic.bundle_search._vector_search", new_callable=AsyncMock) as mock_vec, \
         patch("m_flow.retrieval.episodic.bundle_search.get_episodic_memory_fragment", new_callable=AsyncMock) as mock_proj, \
         patch("m_flow.retrieval.episodic.bundle_search.compute_episode_bundles", return_value=[]), \
         patch("m_flow.retrieval.episodic.bundle_search.get_vector_provider") as mock_vprov:
        
        # Setup mock hit with real float score
        mock_hit = MockHit("ep1", 0.5)
        
        mock_vec.return_value = ({"Episode": [mock_hit]}, {})
        mock_proj.return_value = mock_fragment
        mock_vprov.return_value = MagicMock()
        
        results = await episodic_bundle_search("test query", config=mock_config)
        
        assert results == []
        mock_vec.assert_called_once()
        assert mock_proj.call_count >= 1
