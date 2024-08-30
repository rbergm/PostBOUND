SELECT COUNT(*)
FROM comments AS c,
  posts AS p,
  postLinks AS pl,
  postHistory AS ph,
  votes AS v,
  badges AS b,
  users AS u
WHERE p.Id = pl.RelatedPostId
  AND b.UserId = u.Id
  AND c.UserId = u.Id
  AND p.Id = v.PostId
  AND p.Id = c.PostId
  AND p.Id = ph.PostId
  AND p.Score <= 40
  AND p.CommentCount >= 0
  AND p.CreationDate >= CAST('2010-07-28 17:40:56' AS timestamp)
  AND p.CreationDate <= CAST('2014-09-11 04:22:44' AS timestamp)
  AND pl.LinkTypeId = 1;
