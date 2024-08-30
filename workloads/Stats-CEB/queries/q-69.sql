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
  AND p.Score <= 32
  AND p.ViewCount <= 4146
  AND pl.LinkTypeId = 1
  AND v.CreationDate <= CAST('2014-09-10 00:00:00' AS timestamp);
