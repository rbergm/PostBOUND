SELECT COUNT(*)
FROM comments AS c,
  posts AS p,
  postLinks AS pl,
  postHistory AS ph,
  votes AS v,
  badges AS b
WHERE p.Id = c.PostId
  AND p.Id = pl.RelatedPostId
  AND p.Id = ph.PostId
  AND p.Id = v.PostId
  AND b.UserId = c.UserId
  AND c.Score = 0
  AND p.Score <= 67
  AND ph.PostHistoryTypeId = 34
  AND b.Date <= CAST('2014-08-20 12:16:56' AS timestamp);
